//! Simulates a genetic algorithm on a population in order to improve the fit score and performance. The simulations
//! are performed in a tournament bracket configuration so that populations can compete against each other.

pub mod genetic_node;

use crate::error::Error;
use crate::tree;
use genetic_node::GeneticNodeWrapper;

use file_linked::FileLinked;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
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
/// # #[derive(Default, Deserialize, Serialize, Clone, PartialEq, Debug)]
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
/// #
/// #     fn merge(left: &TestState, right: &TestState) -> Result<Box<Self>, Error> {
/// #         Ok(Box::new(left.clone()))
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
#[derive(Clone, Serialize, Deserialize, Copy, Debug, PartialEq)]
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

/// Creates a tournament style bracket for simulating and evaluating nodes of type `T` implementing [`GeneticNode`].
/// These nodes are built upwards as a balanced binary tree starting from the bottom. This results in `Bracket` building
/// a separate tree of the same height then merging trees together. Evaluating populations between nodes and taking the strongest
/// individuals.
///
/// [`GeneticNode`]: genetic_node::GeneticNode
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Bracket<T>
where
    T: genetic_node::GeneticNode + Serialize,
{
    pub tree: tree::Tree<Option<GeneticNodeWrapper<T>>>,
    iteration_scaling: IterationScaling,
}

impl<T> Bracket<T>
where
    T: genetic_node::GeneticNode
        + Default
        + DeserializeOwned
        + Serialize
        + Clone
        + PartialEq
        + Debug,
{
    /// Initializes a bracket of type `T` storing the contents to `file_path`
    ///
    /// # Examples
    /// ```
    /// # use gemla::bracket::*;
    /// # use gemla::btree;
    /// # use gemla::tree;
    /// # use gemla::error::Error;
    /// # use serde::{Deserialize, Serialize};
    /// # use std::fmt;
    /// # use std::str::FromStr;
    /// # use std::string::ToString;
    /// # use std::path;
    /// #
    /// #[derive(Default, Deserialize, Serialize, Debug, Clone, PartialEq)]
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
    /// #
    /// #     fn merge(left: &TestState, right: &TestState) -> Result<Box<Self>, Error> {
    /// #         Ok(Box::new(left.clone()))
    /// #     }
    /// }
    ///
    /// # fn main() {
    /// let mut bracket = Bracket::<TestState>::initialize(path::PathBuf::from("./temp"))
    /// .expect("Bracket failed to initialize");
    ///
    /// std::fs::remove_file("./temp").expect("Unable to remove file");
    /// # }
    /// ```
    pub fn initialize(file_path: path::PathBuf) -> Result<FileLinked<Self>, Error> {
        Ok(FileLinked::new(
            Bracket {
                tree: btree!(Some(GeneticNodeWrapper::new()?)),
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
    /// # #[derive(Default, Deserialize, Serialize, Clone, PartialEq, Debug)]
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
    /// #
    /// #     fn merge(left: &TestState, right: &TestState) -> Result<Box<Self>, Error> {
    /// #         Ok(Box::new(left.clone()))
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
    fn create_new_branch(
        &self,
        height: u64,
    ) -> Result<tree::Tree<Option<GeneticNodeWrapper<T>>>, Error> {
        if height == 1 {
            let mut base_node = GeneticNodeWrapper::new()?;

            base_node.process_node(match self.iteration_scaling {
                IterationScaling::Linear(x) => x * height,
                IterationScaling::Constant(x) => x,
            })?;

            Ok(btree!(Some(base_node)))
        } else {
            let left = self.create_new_branch(height - 1)?;
            let right = self.create_new_branch(height - 1)?;
            let mut new_val = if left.val.clone().unwrap().data.unwrap().get_fit_score()
                >= right.val.clone().unwrap().data.unwrap().get_fit_score()
            {
                left.val.clone().unwrap()
            } else {
                right.val.clone().unwrap()
            };

            new_val.process_node(match self.iteration_scaling {
                IterationScaling::Linear(x) => x * height,
                IterationScaling::Constant(x) => x,
            })?;

            Ok(btree!(Some(new_val), left, right))
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
    /// # #[derive(Default, Deserialize, Serialize, Clone, PartialEq, Debug)]
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
    /// #
    /// #     fn merge(left: &TestState, right: &TestState) -> Result<Box<Self>, Error> {
    /// #         Ok(Box::new(left.clone()))
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

        self.tree
            .val
            .clone()
            .unwrap()
            .process_node(match self.iteration_scaling {
                IterationScaling::Linear(x) => (x * self.tree.height()),
                IterationScaling::Constant(x) => x,
            })?;

        let new_val = if new_branch
            .val
            .clone()
            .unwrap()
            .data
            .unwrap()
            .get_fit_score()
            >= self.tree.val.clone().unwrap().data.unwrap().get_fit_score()
        {
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
    use crate::bracket::*;
    use crate::tree::*;

    use serde::{Deserialize, Serialize};
    use std::str::FromStr;

    #[derive(Default, Deserialize, Serialize, Clone, Debug, PartialEq)]
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

        fn initialize() -> Result<Box<TestState>, Error> {
            Ok(Box::new(TestState { score: 0.0 }))
        }

        fn merge(left: &TestState, right: &TestState) -> Result<Box<TestState>, Error> {
            Ok(Box::new(if left.get_fit_score() > right.get_fit_score() {
                left.clone()
            } else {
                right.clone()
            }))
        }
    }

    #[test]
    fn test_new() {
        let bracket = Bracket::<TestState>::initialize(path::PathBuf::from("./temp"))
            .expect("Bracket failed to initialize");

        assert_eq!(
            bracket,
            file_linked::FileLinked::new(
                Bracket {
                    tree: Tree {
                        val: Some(GeneticNodeWrapper::new().unwrap()),
                        left: None,
                        right: None
                    },
                    iteration_scaling: IterationScaling::Constant(1)
                },
                path::PathBuf::from("./temp")
            )
            .unwrap()
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

        assert_eq!(bracket.readonly().tree.height(), 4);
        assert_eq!(
            bracket
                .readonly()
                .tree
                .val
                .clone()
                .unwrap()
                .data
                .unwrap()
                .score,
            15.0
        );

        std::fs::remove_file("./temp2").expect("Unable to remove file");
    }
}
