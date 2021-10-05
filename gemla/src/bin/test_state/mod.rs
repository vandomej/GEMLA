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

#[cfg(test)]
mod tests {
    use super::*;
    use gemla::bracket::genetic_node::GeneticNode;

    #[test]
    fn test_initialize() {
        let state = TestState::initialize().unwrap();

        assert_eq!(state.population.len(), POPULATION_SIZE as usize);
    }

    #[test]
    fn test_simulate() {
        let mut state = TestState {
            thread_rng: thread_rng(),
            population: vec![1.0, 1.0, 2.0, 3.0],
        };

        let original_population = state.population.clone();

        state.simulate(0).unwrap();
        assert_eq!(original_population, state.population);

        state.simulate(1).unwrap();
        assert!(original_population
            .iter()
            .zip(state.population.iter())
            .all(|(&a, &b)| b >= a - 10.0 && b <= a + 10.0));

        state.simulate(2).unwrap();
        assert!(original_population
            .iter()
            .zip(state.population.iter())
            .all(|(&a, &b)| b >= a - 30.0 && b <= a + 30.0))
    }

    #[test]
    fn test_calculate_scores_and_trim() {
        let mut state = TestState {
            thread_rng: thread_rng(),
            population: vec![4.0, 1.0, 1.0, 3.0, 2.0],
        };

        state.calculate_scores_and_trim().unwrap();

        assert_eq!(state.population.len(), POPULATION_REDUCTION_SIZE as usize);
        assert!(state.population.iter().any(|&x| x == 4.0));
        assert!(state.population.iter().any(|&x| x == 3.0));
        assert!(state.population.iter().any(|&x| x == 2.0));
    }

    #[test]
    fn test_mutate() {
        let mut state = TestState {
            thread_rng: thread_rng(),
            population: vec![4.0, 3.0, 3.0],
        };

        state.mutate().unwrap();

        assert_eq!(state.population.len(), POPULATION_SIZE as usize);
    }

    #[test]
    fn test_merge() {
        let state1 = TestState {
            thread_rng: thread_rng(),
            population: vec![1.0, 2.0, 4.0, 5.0],
        };

        let state2 = TestState {
            thread_rng: thread_rng(),
            population: vec![0.0, 1.0, 3.0, 7.0],
        };

        let merged_state = TestState::merge(&state1, &state2).unwrap();

        assert_eq!(merged_state.population.len(), POPULATION_SIZE as usize);
        assert!(merged_state.population.iter().any(|&x| x == 7.0));
        assert!(merged_state.population.iter().any(|&x| x == 5.0));
        assert!(merged_state.population.iter().any(|&x| x == 4.0));
    }
}
