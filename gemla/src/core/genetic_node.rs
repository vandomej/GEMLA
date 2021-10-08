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
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone, Copy)]
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

    fn simulate(&mut self) -> Result<(), Error>;

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
pub struct GeneticNodeWrapper<T> {
    pub node: Option<T>,
    state: GeneticState,
    generation: u64,
    pub total_generations: u64,
}

impl<T> GeneticNodeWrapper<T>
where
    T: GeneticNode + Debug,
{
    pub fn new(total_generations: u64) -> Self {
        GeneticNodeWrapper {
            node: None,
            state: GeneticState::Initialize,
            generation: 0,
            total_generations,
        }
    }

    pub fn from(data: T, total_generations: u64) -> Self {
        GeneticNodeWrapper {
            node: Some(data),
            state: GeneticState::Simulate,
            generation: 0,
            total_generations,
        }
    }

    pub fn state(&self) -> &GeneticState {
        &self.state
    }

    pub fn process_node(&mut self) -> Result<GeneticState, Error> {
        match (&self.state, &self.node) {
            (GeneticState::Initialize, _) => {
                self.node = Some(*T::initialize()?);
                self.state = GeneticState::Simulate;
            }
            (GeneticState::Simulate, Some(_)) => {
                self.node
                    .as_mut()
                    .unwrap()
                    .simulate()
                    .with_context(|| format!("Error simulating node: {:?}", self))?;

                self.state = if self.generation >= self.total_generations {
                    GeneticState::Finish
                } else {
                    GeneticState::Mutate
                };
            }
            (GeneticState::Mutate, Some(_)) => {
                self.node
                    .as_mut()
                    .unwrap()
                    .mutate()
                    .with_context(|| format!("Error mutating node: {:?}", self))?;

                self.generation += 1;
                self.state = GeneticState::Simulate;
            }
            (GeneticState::Finish, Some(_)) => (),
            _ => panic!("Error processing node {:?}", self.node),
        }

        Ok(self.state)
    }
}
