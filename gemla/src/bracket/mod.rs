//! Simulates a genetic algorithm on a population in order to improve the fit score and performance. The simulations
//! are performed in a tournament bracket configuration so that populations can compete against each other.

pub mod genetic_node;

use crate::error::Error;
use crate::tree::Tree;
use anyhow::anyhow;
use file_linked::FileLinked;
use genetic_node::{GeneticNode, GeneticNodeWrapper};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fmt::Debug;
use std::fs::File;
use std::io::ErrorKind;
use std::mem::swap;
use std::path::Path;

/// Creates a tournament style bracket for simulating and evaluating nodes of type `T` implementing [`GeneticNode`].
/// These nodes are built upwards as a balanced binary tree starting from the bottom. This results in `Bracket` building
/// a separate tree of the same height then merging trees together. Evaluating populations between nodes and taking the strongest
/// individuals.
///
/// [`GeneticNode`]: genetic_node::GeneticNode
pub struct Gemla<T>
where
    T: Serialize,
{
    data: FileLinked<Option<Tree<Option<GeneticNodeWrapper<T>>>>>,
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
                        FileLinked::new(Some(btree!(None)), path)?
                    } else {
                        FileLinked::from_file(path)?
                    },
                })
            }
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(Gemla {
                data: FileLinked::new(Some(btree!(None)), path)?,
            }),
            Err(error) => Err(Error::IO(error)),
        }
    }

    pub fn simulate(&mut self, steps: u64) -> Result<(), Error> {
        self.data.mutate(|d| Gemla::increase_height(d, steps))?;

        self.data
            .mutate(|d| Gemla::process_tree(d.as_mut().unwrap()))??;

        Ok(())
    }

    fn build_empty_tree(size: usize) -> Tree<Option<GeneticNodeWrapper<T>>> {
        if size <= 1 {
            btree!(None)
        } else {
            btree!(
                None,
                Gemla::build_empty_tree(size - 1),
                Gemla::build_empty_tree(size - 1)
            )
        }
    }

    fn increase_height(tree: &mut Option<Tree<Option<GeneticNodeWrapper<T>>>>, amount: u64) {
        for _ in 0..amount {
            let height = tree.as_ref().unwrap().height();
            let temp = tree.take();
            swap(
                tree,
                &mut Some(btree!(
                    None,
                    temp.unwrap(),
                    Gemla::build_empty_tree(height as usize)
                )),
            );
        }
    }

    fn process_tree(tree: &mut Tree<Option<GeneticNodeWrapper<T>>>) -> Result<(), Error> {
        if tree.val.is_none() {
            match (&mut tree.left, &mut tree.right) {
                (Some(l), Some(r)) => {
                    Gemla::process_tree(&mut (*l))?;
                    Gemla::process_tree(&mut (*r))?;

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
