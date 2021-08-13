pub mod genetic_node;
pub mod genetic_state;

use super::file_linked::FileLinked;
use super::tree;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use std::string::ToString;

#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
#[serde(tag = "enumType", content = "enumContent")]
pub enum IterationScaling {
    Linear(u32),
}

impl Default for IterationScaling {
    fn default() -> Self {
        IterationScaling::Linear(1)
    }
}

impl fmt::Display for IterationScaling {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            serde_json::to_string(self).expect("Unable to deserialize IterationScaling struct")
        )
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Bracket<T> {
    tree: tree::Tree<T>,
    step: u64,
    iteration_scaling: IterationScaling,
}

impl<T: fmt::Display + Serialize> fmt::Display for Bracket<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            serde_json::to_string(self).expect("Unable to deserialize Bracket struct")
        )
    }
}

impl<T> Bracket<T>
where
    T: genetic_node::GeneticNode
        + ToString
        + FromStr
        + Default
        + fmt::Display
        + DeserializeOwned
        + Serialize
        + Clone,
{
    pub fn initialize(file_path: String) -> Result<FileLinked<Self>, String> {
        FileLinked::new(
            Bracket {
                tree: btree!(*T::initialize()?),
                step: 0,
                iteration_scaling: IterationScaling::default(),
            },
            file_path,
        )
    }

    pub fn iteration_scaling(&mut self, iteration_scaling: IterationScaling) -> &mut Self {
        self.iteration_scaling = iteration_scaling;
        self
    }

    pub fn create_new_branch(&self, height: u64) -> Result<tree::Tree<T>, String> {
        if height == 1 {
            let mut base_node = btree!(*T::initialize()?);

            base_node.val.simulate(match self.iteration_scaling {
                IterationScaling::Linear(x) => (x as u64) * height,
            })?;

            Ok(btree!(base_node.val))
        } else {
            let left = self.create_new_branch(height - 1)?;
            let right = self.create_new_branch(height - 1)?;
            let mut new_val = if left.val.get_fit_score() >= right.val.get_fit_score() {
                left.val.clone()
            } else {
                right.val.clone()
            };

            new_val.simulate(match self.iteration_scaling {
                IterationScaling::Linear(x) => (x as u64) * height,
            })?;

            Ok(btree!(new_val, left, right))
        }
    }

    pub fn run_simulation_step(&mut self) -> Result<&mut Self, String> {
        let new_branch = self.create_new_branch(self.step + 1)?;

        self.tree.val.simulate(match self.iteration_scaling {
            IterationScaling::Linear(x) => ((x as u64) * (self.step + 1)),
        })?;

        let new_val = if new_branch.val.get_fit_score() >= self.tree.val.get_fit_score() {
            new_branch.val.clone()
        } else {
            self.tree.val.clone()
        };

        self.tree = btree!(new_val, new_branch, self.tree.clone());

        self.step += 1;

        Ok(self)
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    use serde::{Deserialize, Serialize};
    use std::fmt;
    use std::str::FromStr;
    use std::string::ToString;

    #[derive(Default, Deserialize, Serialize, Clone)]
    struct TestState {
        pub score: f64,
    }

    impl FromStr for TestState {
        type Err = String;

        fn from_str(s: &str) -> Result<TestState, Self::Err> {
            toml::from_str(s).map_err(|_| format!("Unable to parse string {}", s))
        }
    }

    impl fmt::Display for TestState {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "{}", self.score)
        }
    }

    impl TestState {
        fn new(score: f64) -> TestState {
            TestState { score: score }
        }
    }

    impl genetic_node::GeneticNode for TestState {
        fn simulate(&mut self, iterations: u64) -> Result<(), String> {
            self.score += iterations as f64;
            Ok(())
        }

        fn get_fit_score(&self) -> f64 {
            self.score
        }

        fn calculate_scores_and_trim(&mut self) -> Result<(), String> {
            Ok(())
        }

        fn mutate(&mut self) -> Result<(), String> {
            Ok(())
        }

        fn initialize() -> Result<Box<Self>, String> {
            Ok(Box::new(TestState { score: 0.0 }))
        }
    }

    #[test]
    fn test_new() {
        let bracket = Bracket::<TestState>::initialize("./temp".to_string())
            .expect("Bracket failed to initialize");

        assert_eq!(
            format!("{}", bracket),
            format!("{{\"tree\":{},\"step\":0,\"iteration_scaling\":{{\"enumType\":\"Linear\",\"enumContent\":1}}}}", 
            btree!(TestState::new(0.0)))
        );

        std::fs::remove_file("./temp").expect("Unable to remove file");
    }

    #[test]
    fn test_run() {
        let mut bracket = Bracket::<TestState>::initialize("./temp2".to_string())
            .expect("Bracket failed to initialize");

        bracket
            .mutate(|b| drop(b.iteration_scaling(IterationScaling::Linear(2))))
            .expect("Failed to set iteration scaling");
        for _ in 0..3 {
            bracket
                .mutate(|b| drop(b.run_simulation_step()))
                .expect("Failed to run step");
        }

        assert_eq!(
            format!("{}", bracket),
            format!("{{\"tree\":{},\"step\":3,\"iteration_scaling\":{{\"enumType\":\"Linear\",\"enumContent\":2}}}}", 
            btree!(
                TestState::new(12.0),
                btree!(
                    TestState::new(12.0),
                    btree!(TestState::new(6.0),
                        btree!(TestState::new(2.0)),
                        btree!(TestState::new(2.0))),
                    btree!(TestState::new(6.0),
                        btree!(TestState::new(2.0)),
                        btree!(TestState::new(2.0)))
                ),
                btree!(
                    TestState::new(12.0),
                    btree!(TestState::new(6.0),
                        btree!(TestState::new(2.0)),
                        btree!(TestState::new(2.0))),
                    btree!(TestState::new(6.0),
                        btree!(TestState::new(2.0)),
                        btree!(TestState::new(2.0))))
                )
            )
        );

        std::fs::remove_file("./temp2").expect("Unable to remove file");
    }

}