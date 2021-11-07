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
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GeneticNodeWrapper<T> {
    node: Option<T>,
    state: GeneticState,
    generation: u64,
    max_generations: u64,
    id: uuid::Uuid,
}

impl<T> Default for GeneticNodeWrapper<T> {
    fn default() -> Self {
        GeneticNodeWrapper {
            node: None,
            state: GeneticState::Initialize,
            generation: 0,
            max_generations: 1,
            id: uuid::Uuid::new_v4(),
        }
    }
}

impl<T> GeneticNodeWrapper<T>
where
    T: GeneticNode + Debug,
{
    pub fn new(max_generations: u64) -> Self {
        GeneticNodeWrapper::<T> {
            max_generations,
            ..Default::default()
        }
    }

    pub fn from(data: T, max_generations: u64, id: uuid::Uuid) -> Self {
        GeneticNodeWrapper {
            node: Some(data),
            state: GeneticState::Simulate,
            generation: 0,
            max_generations,
            id,
        }
    }

    pub fn as_ref(&self) -> Option<&T> {
        self.node.as_ref()
    }

    pub fn id(&self) -> uuid::Uuid {
        self.id
    }

    pub fn max_generations(&self) -> u64 {
        self.max_generations
    }

    pub fn state(&self) -> GeneticState {
        self.state
    }

    pub fn process_node(&mut self) -> Result<GeneticState, Error> {
        match (self.state, &mut self.node) {
            (GeneticState::Initialize, _) => {
                self.node = Some(*T::initialize()?);
                self.state = GeneticState::Simulate;
            }
            (GeneticState::Simulate, Some(n)) => {
                n.simulate()
                    .with_context(|| format!("Error simulating node: {:?}", self))?;

                self.state = if self.generation >= self.max_generations {
                    GeneticState::Finish
                } else {
                    GeneticState::Mutate
                };
            }
            (GeneticState::Mutate, Some(n)) => {
                n.mutate()
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
