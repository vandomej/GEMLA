pub trait GeneticState {
    fn run_simulation(&mut self, iterations: u32);

    fn get_fit_score(&self) -> f64;

    fn initialize() -> Self;
}