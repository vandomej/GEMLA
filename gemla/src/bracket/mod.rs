//! Simulates a genetic algorithm on a population in order to improve the fit score and performance. The simulations
//! are performed in a tournament bracket configuration so that populations can compete against each other.

pub mod genetic_node;

use crate::error::Error;
use crate::tree::Tree;
use anyhow::anyhow;
use file_linked::FileLinked;
use genetic_node::{GeneticNode, GeneticNodeWrapper};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::fs::File;
use std::io::ErrorKind;
use std::mem::replace;
use std::path::Path;

/// As the bracket tree increases in height, `IterationScaling` can be used to configure the number of iterations that
/// a node runs for.
///
/// # Examples
///
/// TODO
#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
struct Bracket<T>
{
    tree: Option<Tree<Option<GeneticNodeWrapper<T>>>>,
    iteration_scaling: IterationScaling,
}

impl<T> Bracket<T>
where
    T: GeneticNode + Serialize + Debug,
{
    fn build_empty_tree(size: usize) -> Tree<Option<GeneticNodeWrapper<T>>> {
        if size <= 1 {
            btree!(None)
        } else {
            btree!(
                None,
                Bracket::build_empty_tree(size - 1),
                Bracket::build_empty_tree(size - 1)
            )
        }
    }

    fn increase_height(&mut self, amount: u64) {
        for _ in 0..amount {
            let height = self.tree.as_ref().unwrap().height();
            let tree = replace(&mut self.tree, None);
            drop(replace(
                &mut self.tree,
                Some(btree!(
                    None,
                    tree.unwrap(),
                    Bracket::build_empty_tree(height as usize)
                )),
            ));
        }
    }

    fn process_tree(tree: &mut Tree<Option<GeneticNodeWrapper<T>>>) -> Result<(), Error> {
        if tree.val.is_none() {
            match (&mut tree.left, &mut tree.right) {
                (Some(l), Some(r)) => {
                    Bracket::process_tree(&mut (*l))?;
                    Bracket::process_tree(&mut (*r))?;

                    let left_node = (*l).val.as_ref().unwrap().data.as_ref().unwrap();
                    let right_node = (*r).val.as_ref().unwrap().data.as_ref().unwrap();
                    let merged_node = GeneticNode::merge(left_node, right_node)?;

                    tree.val = Some(GeneticNodeWrapper::from(*merged_node)?);
                    tree.val.as_mut().unwrap().process_node(1)?;
                }
                (None, None) => {
                    tree.val = Some(GeneticNodeWrapper::new()?);
                    tree.val.as_mut().unwrap().process_node(1)?;
                }
                _ => {
                    return Err(Error::Other(anyhow!("unable to process tree {:?}", tree)));
                }
            }
        }

        Ok(())
    }

    fn process(&mut self) -> Result<(), Error> {
        Bracket::process_tree(self.tree.as_mut().unwrap())?;

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
where T: Serialize
{
    data: FileLinked<Bracket<T>>,
}

impl<T> Gemla<T>
where
    T: GeneticNode + Serialize + DeserializeOwned + Debug,
{
    pub fn new(path: &Path, overwrite: bool) -> Result<Self, Error> {
        match File::open(path) {
            Ok(file) => {
                drop(file);

                Ok(Gemla {
                    data: if overwrite {
                        FileLinked::new(
                            Bracket {
                                tree: Some(btree!(None)),
                                iteration_scaling: IterationScaling::default(),
                            },
                            path,
                        )?
                    } else {
                        FileLinked::from_file(path)?
                    },
                })
            }
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(Gemla {
                data: FileLinked::new(
                    Bracket {
                        tree: Some(btree!(None)),
                        iteration_scaling: IterationScaling::default(),
                    },
                    path,
                )?,
            }),
            Err(error) => Err(Error::IO(error)),
        }
    }

    pub fn simulate(&mut self, steps: u64) -> Result<(), Error> {
        self.data.mutate(|b| b.increase_height(steps))?;

        self.data.mutate(|b| b.process())??;

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
