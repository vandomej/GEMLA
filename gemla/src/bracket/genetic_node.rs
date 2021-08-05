//! A trait used to interact with the internal state of nodes within the genetic bracket

use super::genetic_state::GeneticState;

use serde::{Deserialize, Serialize};
use std::fmt;

/// A trait used to interact with the internal state of nodes within the genetic bracket
pub trait GeneticNode {
    /// Initializes a new instance of a genetic state.
    fn initialize() -> Result<Box<Self>, String>;

    /// Runs a simulation on the state object in order to guage it's fitness.
    /// - iterations: the number of iterations (learning cycles) that the current state should simulate
    ///
    /// This will be called for every node in a bracket before evaluating it's fitness against other nodes.
    fn simulate(&mut self, iterations: u64) -> Result<(), String>;

    /// Returns a fit score associated with the nodes performance.
    /// This will be used by a bracket in order to determine the most successful child.
    fn get_fit_score(&self) -> f64;

    /// Used when scoring the nodes after simulating and should remove underperforming children.
    fn calculate_scores_and_trim(&mut self) -> Result<(), String>;

    /// Mutates members in a population and/or crossbreeds them to produce new offspring.
    fn mutate(&mut self) -> Result<(), String>;
}

/// Used externally to wrap a node implementing the GeneticNode trait. Processes state transitions for the given node as well as signal recovery.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GeneticNodeWrapper<T>
where
    T: GeneticNode,
{
    data: Option<T>,
    state: GeneticState,
    iteration: u32,
}

impl<T> GeneticNodeWrapper<T>
where
    T: GeneticNode + fmt::Debug,
{
    /// Initializes a wrapper around a GeneticNode
    fn new() -> Result<Self, String> {
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

    fn process_node(&mut self, iterations: u32) -> Result<(), String> {
        let mut result = Ok(());

        loop {
            match (self.state, self.data.as_ref()) {
                (GeneticState::Initialize, _) => {
                    self.iteration = 0;
                    let new_data =
                        T::initialize().map_err(|e| format!("Error initializing node: {}", e))?;
                    self.data = Some(*new_data);
                    self.state = GeneticState::Simulate;
                }
                (GeneticState::Simulate, Some(_)) => {
                    self.data
                        .as_mut()
                        .unwrap()
                        .simulate(5)
                        .map_err(|e| format!("Error simulating node: {}", e))?;
                    self.state = GeneticState::Score;
                }
                (GeneticState::Score, Some(_)) => {
                    self.data
                        .as_mut()
                        .unwrap()
                        .calculate_scores_and_trim()
                        .map_err(|e| format!("Error scoring and trimming node: {}", e))?;

                    self.state = if self.iteration == iterations {
                        GeneticState::Finish
                    } else {
                        GeneticState::Mutate
                    }
                }
                (GeneticState::Mutate, Some(_)) => {
                    self.data
                        .as_mut()
                        .unwrap()
                        .mutate()
                        .map_err(|e| format!("Error mutating node: {}", e))?;
                    self.state = GeneticState::Simulate;
                }
                (GeneticState::Finish, Some(_)) => {
                    break;
                }
                _ => result = Err(format!("Error processing node {:?}", self.data)),
            }
        }

        result
    }
}
