//! Simulates a genetic algorithm on a population in order to improve the fit score and performance. The simulations
//! are performed in a tournament bracket configuration so that populations can compete against each other.

pub mod genetic_node;

use crate::error::Error;
use crate::tree::Tree;
use anyhow::anyhow;
use file_linked::FileLinked;
use genetic_node::{GeneticNode, GeneticNodeWrapper, GeneticState};
use log::{info, trace};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::fs::File;
use std::io::ErrorKind;
use std::mem::swap;
use std::path::Path;
use std::time::Instant;

type SimulationTree<T> = Tree<GeneticNodeWrapper<T>>;

#[derive(Serialize, Deserialize)]
pub struct GemlaConfig {
    pub generations_per_node: u64,
    pub overwrite: bool,
}

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
    pub data: FileLinked<(Option<SimulationTree<T>>, GemlaConfig)>,
}

impl<T> Gemla<T>
where
    T: GeneticNode + Serialize + DeserializeOwned + Debug,
{
    pub fn new(path: &Path, config: GemlaConfig) -> Result<Self, Error> {
        match File::open(path) {
            Ok(file) => {
                drop(file);

                Ok(Gemla {
                    data: if config.overwrite {
                        FileLinked::new((None, config), path)?
                    } else {
                        FileLinked::from_file(path)?
                    },
                })
            }
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(Gemla {
                data: FileLinked::new((None, config), path)?,
            }),
            Err(error) => Err(Error::IO(error)),
        }
    }

    pub fn simulate(&mut self, steps: u64) -> Result<(), Error> {
        self.data
            .mutate(|(d, c)| Gemla::increase_height(d, c, steps))??;

        info!(
            "Height of simulation tree increased to {}",
            self.data.readonly().0.as_ref().unwrap().height()
        );

        loop {
            if Gemla::tree_processed(self.data.readonly().0.as_ref().unwrap())? {
                info!("Processed tree");
                break;
            }

            self.data
                .mutate(|(d, _)| Gemla::process_tree(d.as_mut().unwrap()))??;
        }

        Ok(())
    }

    fn increase_height(
        tree: &mut Option<SimulationTree<T>>,
        config: &GemlaConfig,
        amount: u64,
    ) -> Result<(), Error> {
        for _ in 0..amount {
            if tree.is_none() {
                swap(
                    tree,
                    &mut Some(btree!(GeneticNodeWrapper::new(config.generations_per_node))),
                );
            } else {
                let height = tree.as_mut().unwrap().height() as u64;
                let temp = tree.take();
                swap(
                    tree,
                    &mut Some(btree!(
                        GeneticNodeWrapper::new(config.generations_per_node),
                        temp.unwrap(),
                        btree!(GeneticNodeWrapper::new(
                            height * config.generations_per_node
                        ))
                    )),
                );
            }
        }

        Ok(())
    }

    fn tree_processed(tree: &SimulationTree<T>) -> Result<bool, Error> {
        if tree.val.state() == &GeneticState::Finish {
            match (&tree.left, &tree.right) {
                (Some(l), Some(r)) => Ok(Gemla::tree_processed(l)? && Gemla::tree_processed(r)?),
                (None, None) => Ok(true),
                _ => Err(Error::Other(anyhow!("unable to process tree {:?}", tree))),
            }
        } else {
            Ok(false)
        }
    }

    fn process_tree(tree: &mut SimulationTree<T>) -> Result<(), Error> {
        if tree.val.state() == &GeneticState::Initialize {
            match (&mut tree.left, &mut tree.right) {
                (Some(l), _) if l.val.state() != &GeneticState::Finish => {
                    Gemla::process_tree(&mut (*l))?;
                }
                (_, Some(r)) if r.val.state() != &GeneticState::Finish => {
                    Gemla::process_tree(&mut (*r))?;
                }
                (Some(l), Some(r))
                    if r.val.state() == &GeneticState::Finish
                        && l.val.state() == &GeneticState::Finish =>
                {
                    let left_node = (*l).val.node.as_ref().unwrap();
                    let right_node = (*r).val.node.as_ref().unwrap();
                    let merged_node = GeneticNode::merge(left_node, right_node)?;

                    tree.val = GeneticNodeWrapper::from(*merged_node, tree.val.total_generations);
                    Gemla::process_node(&mut tree.val)?;
                }
                (None, None) => {
                    Gemla::process_node(&mut tree.val)?;
                }
                _ => {
                    return Err(Error::Other(anyhow!("unable to process tree {:?}", tree)));
                }
            }
        } else if tree.val.state() != &GeneticState::Finish {
            Gemla::process_node(&mut tree.val)?;
        }

        Ok(())
    }

    fn process_node(node: &mut GeneticNodeWrapper<T>) -> Result<(), Error> {
        let node_state_time = Instant::now();
        let node_state = *node.state();

        node.process_node()?;

        trace!(
            "{:?} completed in {:?} for",
            node_state,
            node_state_time.elapsed()
        );

        if node.state() == &GeneticState::Finish {
            info!("Processed node");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::core::*;

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
        fn simulate(&mut self) -> Result<(), Error> {
            self.score += 1.0;
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
