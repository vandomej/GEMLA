//! A trait used to interact with the internal state of nodes within the [`Bracket`]
//!
//! [`Bracket`]: crate::bracket::Bracket

use crate::error::Error;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// An enum used to control the state of a [`GeneticNode`]
///
/// [`GeneticNode`]: crate::bracket::genetic_node
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "enumType", content = "enumContent")]
pub enum GeneticState {
    /// The node and it's data have not finished initializing
    Initialize,
    /// The node is currently simulating a round against target data to determine the fitness of the population
    Simulate,
    /// The node is currently mutating members of it's population and breeding new members
    Mutate,
    /// The node has finished processing for a given number of iterations
    Finish,
}

/// A trait used to interact with the internal state of nodes within the [`Bracket`]
///
/// [`Bracket`]: crate::bracket::Bracket
pub trait GeneticNode {
    /// Initializes a new instance of a [`GeneticState`].
    ///
    /// # Examples
    /// TODO
    fn initialize() -> Result<Box<Self>, Error>;

    /// Runs a simulation on the state object for the given number of `iterations` in order to guage it's fitness.
    /// This will be called for every node in a bracket before evaluating it's fitness against other nodes.
    ///
    /// # Examples
    /// TODO
    fn simulate(&mut self, iterations: u64) -> Result<(), Error>;

    /// Mutates members in a population and/or crossbreeds them to produce new offspring.
    ///
    /// # Examples
    /// TODO
    fn mutate(&mut self) -> Result<(), Error>;

    fn merge(left: &Self, right: &Self) -> Result<Box<Self>, Error>;
}

/// Used externally to wrap a node implementing the [`GeneticNode`] trait. Processes state transitions for the given node as
/// well as signal recovery. Transition states are given by [`GeneticState`]
#[derive(Debug, Serialize, Deserialize)]
pub struct GeneticNodeWrapper<T>
{
    pub data: Option<T>,
    state: GeneticState,
    pub iteration: u64,
}

impl<T> GeneticNodeWrapper<T>
where
    T: GeneticNode + Debug,
{
    /// Initializes a wrapper around a GeneticNode. If the initialization is successful the internal state will be changed to
    /// `GeneticState::Simulate` otherwise it will remain as `GeneticState::Initialize` and will attempt to be created in
    /// [`process_node`](#method.process_node).
    ///
    /// # Examples
    /// TODO
    pub fn new() -> Result<Self, Error> {
        let mut node = GeneticNodeWrapper {
            data: None,
            state: GeneticState::Initialize,
            iteration: 0,
        };

        let new_data = T::initialize()?;
        node.data = Some(*new_data);
        node.state = GeneticState::Simulate;

        Ok(node)
    }

    pub fn from(data: T) -> Result<Self, Error> {
        let mut node = GeneticNodeWrapper {
            data: Some(data),
            state: GeneticState::Initialize,
            iteration: 0,
        };

        node.state = GeneticState::Simulate;

        Ok(node)
    }

    /// Performs state transitions on the [`GeneticNode`] wrapped by the [`GeneticNodeWrapper`].
    /// Will loop through the node training and scoring process for the given number of `iterations`.
    ///
    /// ## Transitions
    /// - `GeneticState::Initialize`: will attempt to call [`initialize`] on the node. When done successfully will change
    ///     the state to `GeneticState::Simulate`
    /// - `GeneticState::Simulate`: Will call [`simulate`] with a number of iterations (not for `iterations`). Will change the state to `GeneticState::Score`
    /// - `GeneticState::Mutate`: Will call [`mutate`] and will change the state to `GeneticState::Simulate.`
    /// - `GeneticState::Finish`: Will finish processing the node and return.
    ///
    /// [`initialize`]: crate::bracket::genetic_node::GeneticNode#tymethod.initialize
    /// [`simulate`]: crate::bracket::genetic_node::GeneticNode#tymethod.simulate
    /// [`mutate`]: crate::bracket::genetic_node::GeneticNode#tymethod.mutate
    pub fn process_node(&mut self, iterations: u64) -> Result<(), Error> {
        // Looping through each state transition until the number of iterations have been reached.
        loop {
            match (&self.state, &self.data) {
                (GeneticState::Initialize, _) => {
                    self.iteration = 0;
                    let new_data = T::initialize()
                        .with_context(|| format!("Error initializing node {:?}", self))?;
                    self.data = Some(*new_data);
                    self.state = GeneticState::Simulate;
                }
                (GeneticState::Simulate, Some(_)) => {
                    self.data
                        .as_mut()
                        .unwrap()
                        .simulate(5)
                        .with_context(|| format!("Error simulating node: {:?}", self))?;

                    self.state = if self.iteration == iterations {
                        GeneticState::Finish
                    } else {
                        GeneticState::Mutate
                    };
                }
                (GeneticState::Mutate, Some(_)) => {
                    self.data
                        .as_mut()
                        .unwrap()
                        .mutate()
                        .with_context(|| format!("Error mutating node: {:?}", self))?;

                    self.iteration += 1;
                    self.state = GeneticState::Simulate;
                }
                (GeneticState::Finish, Some(_)) => {
                    break;
                }
                _ => panic!("Error processing node {:?}", self.data),
            }
        }

        Ok(())
    }
}
