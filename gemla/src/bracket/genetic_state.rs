//! A trait used to interact with the internal state of nodes within the genetic bracket

/// A trait used to interact with the internal state of nodes within the genetic bracket
pub trait GeneticState {
    
    /// Runs a simulation on the state object in order to guage it's fitness.
    /// - iterations: the number of iterations (learning cycles) that the current state should simulate
    /// 
    /// This will be called for every node in a bracket before evaluating it's fitness against other nodes.
    fn run_simulation(&mut self, iterations: u32);

    /// Returns a fit score associated with the nodes performance.
    /// This will be used by a bracket in order to determine the most successful child.
    fn get_fit_score(&self) -> f64;

    /// Initializes a new instance of a genetic state.
    fn initialize() -> Self;
}