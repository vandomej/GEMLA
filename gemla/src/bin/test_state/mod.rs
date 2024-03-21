use gemla::{core::genetic_node::{GeneticNode, GeneticNodeContext}, error::Error};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use async_trait::async_trait;

const POPULATION_SIZE: u64 = 5;
const POPULATION_REDUCTION_SIZE: u64 = 3;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TestState {
    pub population: Vec<i64>,
}

#[async_trait]
impl GeneticNode for TestState {
    fn initialize(_context: GeneticNodeContext) -> Result<Box<Self>, Error> {
        let mut population: Vec<i64> = vec![];

        for _ in 0..POPULATION_SIZE {
            population.push(thread_rng().gen_range(0..100))
        }

        Ok(Box::new(TestState { population }))
    }

    async fn simulate(&mut self, _context: GeneticNodeContext) -> Result<(), Error> {
        let mut rng = thread_rng();

        self.population = self
            .population
            .iter()
            .map(|p| p.saturating_add(rng.gen_range(-1..2)))
            .collect();

        Ok(())
    }

    fn mutate(&mut self, _context: GeneticNodeContext) -> Result<(), Error> {
        let mut rng = thread_rng();

        let mut v = self.population.clone();

        v.sort_unstable();
        v.reverse();

        self.population = v[0..(POPULATION_REDUCTION_SIZE as usize)].to_vec();

        loop {
            if self.population.len() as u64 >= POPULATION_SIZE {
                break;
            }

            let new_individual_index = rng.gen_range(0..self.population.len());
            let mut cross_breed_index = rng.gen_range(0..self.population.len());

            loop {
                if new_individual_index != cross_breed_index {
                    break;
                }

                cross_breed_index = rng.gen_range(0..self.population.len());
            }

            let mut new_individual = self.population.clone()[new_individual_index];
            let cross_breed = self.population.clone()[cross_breed_index];

            new_individual = (new_individual.saturating_add(cross_breed) / 2)
                .saturating_add(rng.gen_range(-1..2));

            self.population.push(new_individual);
        }

        Ok(())
    }

    fn merge(left: &TestState, right: &TestState, id: &Uuid) -> Result<Box<TestState>, Error> {
        let mut v = left.population.clone();
        v.append(&mut right.population.clone());

        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.reverse();

        v = v[..(POPULATION_REDUCTION_SIZE as usize)].to_vec();

        let mut result = TestState { population: v };

        result.mutate(GeneticNodeContext {
            id: id.clone(),
            generation: 0,
            max_generations: 0,
            semaphore: None,
        })?;

        Ok(Box::new(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gemla::core::genetic_node::GeneticNode;

    #[test]
    fn test_initialize() {
        let state = TestState::initialize(
            GeneticNodeContext {
                id: Uuid::new_v4(),
                generation: 0,
                max_generations: 0,
                semaphore: None,
            }
        ).unwrap();

        assert_eq!(state.population.len(), POPULATION_SIZE as usize);
    }

    #[tokio::test]
    async fn test_simulate() {
        let mut state = TestState {
            population: vec![1, 1, 2, 3],
        };

        let original_population = state.population.clone();

        state.simulate(
            GeneticNodeContext {
                id: Uuid::new_v4(),
                generation: 0,
                max_generations: 0,
                semaphore: None,
            }
        ).await.unwrap();
        assert!(original_population
            .iter()
            .zip(state.population.iter())
            .all(|(&a, &b)| b >= a - 1 && b <= a + 2));

        state.simulate(
            GeneticNodeContext {
                id: Uuid::new_v4(),
                generation: 0,
                max_generations: 0,
                semaphore: None,
            }
        ).await.unwrap();
        state.simulate(
            GeneticNodeContext {
                id: Uuid::new_v4(),
                generation: 0,
                max_generations: 0,
                semaphore: None,
            }
        ).await.unwrap();
        assert!(original_population
            .iter()
            .zip(state.population.iter())
            .all(|(&a, &b)| b >= a - 3 && b <= a + 6))
    }

    #[test]
    fn test_mutate() {
        let mut state = TestState {
            population: vec![4, 3, 3],
        };

        state.mutate(
            GeneticNodeContext {
                id: Uuid::new_v4(),
                generation: 0,
                max_generations: 0,
                semaphore: None,
            }
        ).unwrap();

        assert_eq!(state.population.len(), POPULATION_SIZE as usize);
    }

    #[test]
    fn test_merge() {
        let state1 = TestState {
            population: vec![1, 2, 4, 5],
        };

        let state2 = TestState {
            population: vec![0, 1, 3, 7],
        };

        let merged_state = TestState::merge(&state1, &state2, &Uuid::new_v4()).unwrap();

        assert_eq!(merged_state.population.len(), POPULATION_SIZE as usize);
        assert!(merged_state.population.iter().any(|&x| x == 7));
        assert!(merged_state.population.iter().any(|&x| x == 5));
        assert!(merged_state.population.iter().any(|&x| x == 4));
    }
}
