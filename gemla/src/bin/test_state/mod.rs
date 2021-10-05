use gemla::bracket::genetic_node::GeneticNode;
use gemla::error;
use rand::prelude::*;
use rand::rngs::ThreadRng;
use std::convert::TryInto;

const POPULATION_SIZE: u64 = 5;
const POPULATION_REDUCTION_SIZE: u64 = 3;

struct TestState {
    pub population: Vec<f64>,
    thread_rng: ThreadRng,
}

impl GeneticNode for TestState {
    fn initialize() -> Result<Box<Self>, error::Error> {
        let mut thread_rng = thread_rng();
        let mut population: Vec<f64> = vec![];

        for _ in 0..POPULATION_SIZE {
            population.push(thread_rng.gen::<u64>() as f64)
        }

        Ok(Box::new(TestState {
            population,
            thread_rng,
        }))
    }

    fn simulate(&mut self, iterations: u64) -> Result<(), error::Error> {
        for _ in 0..iterations {
            self.population = self
                .population
                .clone()
                .iter()
                .map(|p| p + self.thread_rng.gen_range(-10.0..10.0))
                .collect()
        }

        Ok(())
    }

    fn get_fit_score(&self) -> f64 {
        self.population
            .clone()
            .into_iter()
            .reduce(f64::max)
            .unwrap()
    }

    fn calculate_scores_and_trim(&mut self) -> Result<(), error::Error> {
        let mut v = self.population.clone();

        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.reverse();

        self.population = v[0..(POPULATION_REDUCTION_SIZE as usize)].to_vec();

        Ok(())
    }

    fn mutate(&mut self) -> Result<(), error::Error> {
        loop {
            if self.population.len() >= POPULATION_SIZE.try_into().unwrap() {
                break;
            }

            let new_individual_index = self.thread_rng.gen_range(0..self.population.len());
            let mut cross_breed_index = self.thread_rng.gen_range(0..self.population.len());

            loop {
                if new_individual_index != cross_breed_index {
                    break;
                }

                cross_breed_index = self.thread_rng.gen_range(0..self.population.len());
            }

            let mut new_individual = self.population.clone()[new_individual_index];
            let cross_breed = self.population.clone()[cross_breed_index];

            new_individual += cross_breed + self.thread_rng.gen_range(-10.0..10.0);

            self.population.push(new_individual);
        }

        Ok(())
    }

    fn merge(left: &TestState, right: &TestState) -> Result<Box<TestState>, error::Error> {
        let mut v = left.population.clone();
        v.append(&mut right.population.clone());

        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.reverse();

        v = v[..(POPULATION_REDUCTION_SIZE as usize)].to_vec();

        let mut result = TestState {
            population: v,
            thread_rng: thread_rng(),
        };

        result.mutate()?;

        Ok(Box::new(result))
    }
}
