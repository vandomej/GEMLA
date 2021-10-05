//! Simulates a genetic algorithm on a population in order to improve the fit score and performance. The simulations
//! are performed in a tournament bracket configuration so that populations can compete against each other.

pub mod genetic_node;

use crate::error::Error;
use crate::tree;
use genetic_node::{GeneticNodeWrapper, GeneticNode};
use file_linked::FileLinked;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::path::PathBuf;
use std::fs::File;
use std::io::ErrorKind;

/// As the bracket tree increases in height, `IterationScaling` can be used to configure the number of iterations that
/// a node runs for.
///
/// # Examples
///
/// TODO
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

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
struct Bracket<T>
where
    T: GeneticNode + Serialize,
{
    pub tree: tree::Tree<Option<GeneticNodeWrapper<T>>>,
    iteration_scaling: IterationScaling,
}

impl<T> Bracket<T>
where T: GeneticNode + Serialize
{
    fn increase_height(&mut self, _amount: usize) -> Result<(), Error> {
        Ok(())
    }

    fn process_tree(&mut self) -> Result<(), Error> {
        Ok(())
    }
}

/// Creates a tournament style bracket for simulating and evaluating nodes of type `T` implementing [`GeneticNode`].
/// These nodes are built upwards as a balanced binary tree starting from the bottom. This results in `Bracket` building
/// a separate tree of the same height then merging trees together. Evaluating populations between nodes and taking the strongest
/// individuals.
///
/// [`GeneticNode`]: genetic_node::GeneticNode
pub struct Gemla<T>
where T: GeneticNode + Serialize + DeserializeOwned
{
    data: FileLinked<Bracket<T>>
}

impl<T> Gemla<T> 
where
    T: GeneticNode
        + Serialize
        + DeserializeOwned
        + Default
{
    pub fn new(path: &PathBuf, overwrite: bool) -> Result<Self, Error> {
        match File::open(path) {
            Ok(file) => {
                drop(file);

                Ok(Gemla {
                    data: 
                        if overwrite {
                            FileLinked::from_file(path)?
                        } else {
                            FileLinked::new(Bracket {
                                tree: btree!(None),
                                iteration_scaling: IterationScaling::default()
                            }, path)?
                        }
                })
            },
            Err(error) if error.kind() == ErrorKind::NotFound => {
                Ok(Gemla {
                    data: FileLinked::new(Bracket {
                        tree: btree!(None),
                        iteration_scaling: IterationScaling::default()
                    }, path)?
                })
            },
            Err(error) => Err(Error::IO(error))
        }
    }

    pub fn simulate(&mut self, steps: u64) -> Result<(), Error> {
        self.data.mutate(|b| b.increase_height(steps as usize))??;

        self.data.mutate(|b| b.process_tree())??;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::bracket::*;

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
            Ok(Box::new(if left.score > right.score {
                left.clone()
            } else {
                right.clone()
            }))
        }
    }
}
