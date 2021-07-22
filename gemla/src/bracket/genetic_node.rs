//! A trait used to interact with the internal state of nodes within the genetic bracket

use super::genetic_state::GeneticState;

use serde::{Deserialize, Serialize};
use std::fmt;

/// A trait used to interact with the internal state of nodes within the genetic bracket
pub trait GeneticNode {
    /// Runs a simulation on the state object in order to guage it's fitness.
    /// - iterations: the number of iterations (learning cycles) that the current state should simulate
    ///
    /// This will be called for every node in a bracket before evaluating it's fitness against other nodes.
    fn simulate(&mut self, iterations: u64);

    /// Returns a fit score associated with the nodes performance.
    /// This will be used by a bracket in order to determine the most successful child.
    fn get_fit_score(&self) -> f64;

    fn calculate_scores_and_trim(&mut self);

    fn mutate(&mut self);

    /// Initializes a new instance of a genetic state.
    fn initialize() -> Self;
}

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
    fn new() -> Self {
        let mut node = GeneticNodeWrapper {
            data: None,
            state: GeneticState::Initialize,
            iteration: 0,
        };

        node.data = Some(T::initialize());
        node.state = GeneticState::Simulate;

        node
    }

    fn process_node(&mut self, iterations: u32) -> Result<(), String> {
        let mut result = Ok(());

        loop {
            match (self.state, self.data.as_ref()) {
                (GeneticState::Initialize, _) => {
                    self.iteration = 0;
                    self.data = Some(T::initialize());
                    self.state = GeneticState::Simulate;
                }
                (GeneticState::Simulate, Some(_)) => {
                    self.data.as_mut().unwrap().simulate(5);
                    self.state = GeneticState::Score;
                }
                (GeneticState::Score, Some(_)) => {
                    self.data.as_mut().unwrap().calculate_scores_and_trim();

                    self.state = if self.iteration == iterations {
                        GeneticState::Finish
                    } else {
                        GeneticState::Mutate
                    }
                }
                (GeneticState::Mutate, Some(_)) => {
                    self.data.as_mut().unwrap().mutate();
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
