//! Simulates a genetic algorithm on a population in order to improve the fit score and performance. The simulations
//! are performed in a tournament bracket configuration so that populations can compete against each other.

pub mod genetic_node;

use crate::error::Error;
use crate::tree;

use file_linked::FileLinked;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path;

/// As the bracket tree increases in height, `IterationScaling` can be used to configure the number of iterations that
/// a node runs for.
///
/// # Examples
///
/// ```
/// # use gemla::bracket::*;
/// # use gemla::error::Error;
/// # use serde::{Deserialize, Serialize};
/// # use std::fmt;
/// # use std::str::FromStr;
/// # use std::string::ToString;
/// # use std::path;
/// #
/// # #[derive(Default, Deserialize, Serialize, Clone)]
/// # struct TestState {
/// #   pub score: f64,
/// # }
/// #
/// # impl TestState {
/// #     fn new(score: f64) -> TestState {
/// #         TestState { score: score }
/// #     }
/// # }
/// #
/// # impl genetic_node::GeneticNode for TestState {
/// #     fn simulate(&mut self, iterations: u64) -> Result<(), Error> {
/// #         self.score += iterations as f64;
/// #         Ok(())
/// #     }
/// #
/// #     fn get_fit_score(&self) -> f64 {
/// #         self.score
/// #     }
/// #
/// #     fn calculate_scores_and_trim(&mut self) -> Result<(), Error> {
/// #         Ok(())
/// #     }
/// #
/// #     fn mutate(&mut self) -> Result<(), Error> {
/// #         Ok(())
/// #     }
/// #
/// #     fn initialize() -> Result<Box<Self>, Error> {
/// #         Ok(Box::new(TestState { score: 0.0 }))
/// #     }
/// # }
/// #
/// # fn main() {
/// let mut bracket = Bracket::<TestState>::initialize(path::PathBuf::from("./temp"))
/// .expect("Bracket failed to initialize");
///
/// // Constant iteration scaling ensures that every node is simulated 5 times.
/// bracket
///     .mutate(|b| drop(b.iteration_scaling(IterationScaling::Constant(5))))
///     .expect("Failed to set iteration scaling");
///
/// # std::fs::remove_file("./temp").expect("Unable to remove file");
/// # }
/// ```
#[derive(Clone, Serialize, Deserialize, Copy)]
#[serde(tag = "enumType", content = "enumContent")]
pub enum IterationScaling {
    /// Scales the number of simulations linearly with the height of the  bracket tree given by `f(x) = mx` where
    /// x is the height and m is the linear constant provided.
    Linear(u64),
    /// Each node in a bracket is simulated the same number of times.
    Constant(u64),
}

impl Default for IterationScaling {
    fn default() -> Self {
        IterationScaling::Constant(1)
    }
}

impl fmt::Debug for IterationScaling {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            serde_json::to_string(self).expect("Unable to deserialize IterationScaling struct")
        )
    }
}

/// Creates a tournament style bracket for simulating and evaluating nodes of type `T` implementing [`GeneticNode`].
/// These nodes are built upwards as a balanced binary tree starting from the bottom. This results in `Bracket` building
/// a separate tree of the same height then merging trees together. Evaluating populations between nodes and taking the strongest
/// individuals.
///
/// [`GeneticNode`]: genetic_node::GeneticNode
#[derive(Serialize, Deserialize, Clone)]
pub struct Bracket<T>
where
    T: genetic_node::GeneticNode,
{
    pub tree: tree::Tree<T>,
    iteration_scaling: IterationScaling,
}

impl<T> fmt::Debug for Bracket<T>
where
    T: genetic_node::GeneticNode + Serialize,
{
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
    T: genetic_node::GeneticNode + Default + DeserializeOwned + Serialize + Clone,
{
    /// Initializes a bracket of type `T` storing the contents to `file_path`
    ///
    /// # Examples
    /// ```
    /// # use gemla::bracket::*;
    /// # use gemla::btree;
    /// # use gemla::error::Error;
    /// # use serde::{Deserialize, Serialize};
    /// # use std::fmt;
    /// # use std::str::FromStr;
    /// # use std::string::ToString;
    /// # use std::path;
    /// #
    /// #[derive(Default, Deserialize, Serialize, Debug, Clone)]
    /// struct TestState {
    ///   pub score: f64,
    /// }
    ///
    /// # impl FromStr for TestState {
    /// #   type Err = String;
    /// #
    /// #   fn from_str(s: &str) -> Result<TestState, Self::Err> {
    /// #       serde_json::from_str(s).map_err(|_| format!("Unable to parse string {}", s))
    /// #   }
    /// # }
    /// #
    /// # impl fmt::Display for TestState {
    /// #     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    /// #         write!(f, "{}", self.score)
    /// #     }
    /// # }
    /// #
    /// impl TestState {
    ///     fn new(score: f64) -> TestState {
    ///         TestState { score: score }
    ///     }
    /// }
    ///
    /// impl genetic_node::GeneticNode for TestState {
    /// #     fn simulate(&mut self, iterations: u64) -> Result<(), Error> {
    /// #         self.score += iterations as f64;
    /// #         Ok(())
    /// #     }
    /// #
    /// #     fn get_fit_score(&self) -> f64 {
    /// #         self.score
    /// #     }
    /// #
    /// #     fn calculate_scores_and_trim(&mut self) -> Result<(), Error> {
    /// #         Ok(())
    /// #     }
    /// #
    /// #     fn mutate(&mut self) -> Result<(), Error> {
    /// #         Ok(())
    /// #     }
    /// #
    ///     fn initialize() -> Result<Box<Self>, Error> {
    ///         Ok(Box::new(TestState { score: 0.0 }))
    ///     }
    ///
    ///     //...
    /// }
    ///
    /// # fn main() {
    /// let mut bracket = Bracket::<TestState>::initialize(path::PathBuf::from("./temp"))
    /// .expect("Bracket failed to initialize");
    ///
    /// assert_eq!(
    ///    format!("{:?}", bracket),
    ///    format!("{{\"tree\":{:?},\"iteration_scaling\":{{\"enumType\":\"Constant\",\"enumContent\":1}}}}",
    ///    btree!(TestState{score: 0.0}))
    /// );
    ///
    /// std::fs::remove_file("./temp").expect("Unable to remove file");
    /// # }
    /// ```
    pub fn initialize(file_path: path::PathBuf) -> Result<FileLinked<Self>, Error> {
        Ok(FileLinked::new(
            Bracket {
                tree: btree!(*T::initialize()?),
                iteration_scaling: IterationScaling::default(),
            },
            file_path,
        )?)
    }

    /// Given a bracket object, configures it's [`IterationScaling`].
    ///
    /// # Examples
    /// ```
    /// # use gemla::bracket::*;
    /// # use gemla::error::Error;
    /// # use serde::{Deserialize, Serialize};
    /// # use std::fmt;
    /// # use std::str::FromStr;
    /// # use std::string::ToString;
    /// # use std::path;
    /// #
    /// # #[derive(Default, Deserialize, Serialize, Clone)]
    /// # struct TestState {
    /// #   pub score: f64,
    /// # }
    /// #
    /// # impl fmt::Display for TestState {
    /// #     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    /// #         write!(f, "{}", self.score)
    /// #     }
    /// # }
    /// #
    /// # impl TestState {
    /// #     fn new(score: f64) -> TestState {
    /// #         TestState { score: score }
    /// #     }
    /// # }
    /// #
    /// # impl genetic_node::GeneticNode for TestState {
    /// #     fn simulate(&mut self, iterations: u64) -> Result<(), Error> {
    /// #         self.score += iterations as f64;
    /// #         Ok(())
    /// #     }
    /// #
    /// #     fn get_fit_score(&self) -> f64 {
    /// #         self.score
    /// #     }
    /// #
    /// #     fn calculate_scores_and_trim(&mut self) -> Result<(), Error> {
    /// #         Ok(())
    /// #     }
    /// #
    /// #     fn mutate(&mut self) -> Result<(), Error> {
    /// #         Ok(())
    /// #     }
    /// #
    /// #     fn initialize() -> Result<Box<Self>, Error> {
    /// #         Ok(Box::new(TestState { score: 0.0 }))
    /// #     }
    /// # }
    /// #
    /// # fn main() {
    /// let mut bracket = Bracket::<TestState>::initialize(path::PathBuf::from("./temp"))
    /// .expect("Bracket failed to initialize");
    ///
    /// // Constant iteration scaling ensures that every node is simulated 5 times.
    /// bracket
    ///     .mutate(|b| drop(b.iteration_scaling(IterationScaling::Constant(5))))
    ///     .expect("Failed to set iteration scaling");
    ///
    /// # std::fs::remove_file("./temp").expect("Unable to remove file");
    /// # }
    /// ```
    pub fn iteration_scaling(&mut self, iteration_scaling: IterationScaling) -> &mut Self {
        self.iteration_scaling = iteration_scaling;
        self
    }

    // Creates a balanced tree with the given `height` that will be used as a branch of the primary tree.
    // This additionally simulates and evaluates nodes in the branch as it is built.
    fn create_new_branch(&self, height: u64) -> Result<tree::Tree<T>, Error> {
        if height == 1 {
            let mut base_node = btree!(*T::initialize()?);

            base_node.val.simulate(match self.iteration_scaling {
                IterationScaling::Linear(x) => x * height,
                IterationScaling::Constant(x) => x,
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
                IterationScaling::Linear(x) => x * height,
                IterationScaling::Constant(x) => x,
            })?;

            Ok(btree!(new_val, left, right))
        }
    }

    /// Runs one step of simulation on the current bracket which includes:
    /// 1) Creating a new branch of the same height and performing the same steps for each subtree.
    /// 2) Simulating the top node of the current branch.
    /// 3) Comparing the top node of the current branch to the top node of the new branch.
    /// 4) Takes the best performing node and makes it the root of the tree.
    ///
    /// # Examples
    /// ```
    /// # use gemla::bracket::*;
    /// # use gemla::error::Error;
    /// # use serde::{Deserialize, Serialize};
    /// # use std::fmt;
    /// # use std::str::FromStr;
    /// # use std::string::ToString;
    /// # use std::path;
    /// #
    /// # #[derive(Default, Deserialize, Serialize, Clone)]
    /// # struct TestState {
    /// #   pub score: f64,
    /// # }
    /// #
    /// # impl fmt::Display for TestState {
    /// #     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    /// #         write!(f, "{}", self.score)
    /// #     }
    /// # }
    /// #
    /// # impl TestState {
    /// #     fn new(score: f64) -> TestState {
    /// #         TestState { score: score }
    /// #     }
    /// # }
    /// #
    /// # impl genetic_node::GeneticNode for TestState {
    /// #     fn simulate(&mut self, iterations: u64) -> Result<(), Error> {
    /// #         self.score += iterations as f64;
    /// #         Ok(())
    /// #     }
    /// #
    /// #     fn get_fit_score(&self) -> f64 {
    /// #         self.score
    /// #     }
    /// #
    /// #     fn calculate_scores_and_trim(&mut self) -> Result<(), Error> {
    /// #         Ok(())
    /// #     }
    /// #
    /// #     fn mutate(&mut self) -> Result<(), Error> {
    /// #         Ok(())
    /// #     }
    /// #
    /// #     fn initialize() -> Result<Box<Self>, Error> {
    /// #         Ok(Box::new(TestState { score: 0.0 }))
    /// #     }
    /// # }
    /// #
    /// # fn main() {
    /// let mut bracket = Bracket::<TestState>::initialize(path::PathBuf::from("./temp"))
    ///     .expect("Bracket failed to initialize");
    ///
    /// // Running simulations 3 times
    /// for _ in 0..3 {
    ///     bracket
    ///         .mutate(|b| drop(b.run_simulation_step()))
    ///         .expect("Failed to run step");
    /// }
    ///
    /// assert_eq!(bracket.readonly().tree.height(), 4);
    ///
    /// # std::fs::remove_file("./temp").expect("Unable to remove file");
    /// # }
    /// ```
    pub fn run_simulation_step(&mut self) -> Result<&mut Self, Error> {
        let new_branch = self.create_new_branch(self.tree.height())?;

        self.tree.val.simulate(match self.iteration_scaling {
            IterationScaling::Linear(x) => (x * self.tree.height()),
            IterationScaling::Constant(x) => x,
        })?;

        let new_val = if new_branch.val.get_fit_score() >= self.tree.val.get_fit_score() {
            new_branch.val.clone()
        } else {
            self.tree.val.clone()
        };

        self.tree = btree!(new_val, new_branch, self.tree.clone());

        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde::{Deserialize, Serialize};
    use std::str::FromStr;

    #[derive(Default, Deserialize, Serialize, Clone, Debug)]
    struct TestState {
        pub score: f64,
    }

    impl FromStr for TestState {
        type Err = String;

        fn from_str(s: &str) -> Result<TestState, Self::Err> {
            serde_json::from_str(s).map_err(|_| format!("Unable to parse string {}", s))
        }
    }

    impl genetic_node::GeneticNode for TestState {
        fn simulate(&mut self, iterations: u64) -> Result<(), Error> {
            self.score += iterations as f64;
            Ok(())
        }

        fn get_fit_score(&self) -> f64 {
            self.score
        }

        fn calculate_scores_and_trim(&mut self) -> Result<(), Error> {
            Ok(())
        }

        fn mutate(&mut self) -> Result<(), Error> {
            Ok(())
        }

        fn initialize() -> Result<Box<Self>, Error> {
            Ok(Box::new(TestState { score: 0.0 }))
        }
    }

    #[test]
    fn test_new() {
        let bracket = Bracket::<TestState>::initialize(path::PathBuf::from("./temp"))
            .expect("Bracket failed to initialize");

        assert_eq!(
            format!("{:?}", bracket),
            format!("{{\"tree\":{:?},\"iteration_scaling\":{{\"enumType\":\"Constant\",\"enumContent\":1}}}}", 
            btree!(TestState{score: 0.0}))
        );

        std::fs::remove_file("./temp").expect("Unable to remove file");
    }

    #[test]
    fn test_run() {
        let mut bracket = Bracket::<TestState>::initialize(path::PathBuf::from("./temp2"))
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
            format!("{:?}", bracket),
            format!("{{\"tree\":{:?},\"iteration_scaling\":{{\"enumType\":\"Linear\",\"enumContent\":2}}}}", 
            btree!(
                TestState{score: 12.0},
                btree!(
                    TestState{score: 12.0},
                    btree!(TestState{score: 6.0},
                        btree!(TestState{score: 2.0}),
                        btree!(TestState{score: 2.0})),
                    btree!(TestState{score: 6.0},
                        btree!(TestState{score: 2.0}),
                        btree!(TestState{score: 2.0}))
                ),
                btree!(
                    TestState{score: 12.0},
                    btree!(TestState{score: 6.0},
                        btree!(TestState{score: 2.0}),
                        btree!(TestState{score: 2.0})),
                    btree!(TestState{score: 6.0},
                        btree!(TestState{score: 2.0}),
                        btree!(TestState{score: 2.0})))
                )
            )
        );

        std::fs::remove_file("./temp2").expect("Unable to remove file");
    }
}
