//! Simulates a genetic algorithm on a population in order to improve the fit score and performance. The simulations
//! are performed in a tournament bracket configuration so that populations can compete against each other.

pub mod genetic_node;

use crate::error::Error;
use crate::tree::Tree;
use anyhow::anyhow;
use file_linked::FileLinked;
use genetic_node::{GeneticNode, GeneticNodeWrapper, GeneticState};
use log::{info, trace, warn};
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
pub struct Gemla<'a, T>
where
    T: Serialize + Clone,
{
    pub data: FileLinked<(Option<SimulationTree<T>>, GemlaConfig)>,
    threads: std::collections::HashMap<
        uuid::Uuid,
        futures::prelude::future::BoxFuture<'a, Result<GeneticNodeWrapper<T>, Error>>,
    >,
}

impl<'a, T: 'a> Gemla<'a, T>
where
    T: GeneticNode + Serialize + DeserializeOwned + Debug + Clone + std::marker::Send,
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
                    threads: std::collections::HashMap::new(),
                })
            }
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(Gemla {
                data: FileLinked::new((None, config), path)?,
                threads: std::collections::HashMap::new(),
            }),
            Err(error) => Err(Error::IO(error)),
        }
    }

    pub async fn simulate(&mut self, steps: u64) -> Result<(), Error> {
        self.data
            .mutate(|(d, c)| Gemla::increase_height(d, c, steps))??;

        info!(
            "Height of simulation tree increased to {}",
            self.data.readonly().0.as_ref().unwrap().height()
        );

        loop {
            if Gemla::tree_processed(self.data.readonly().0.as_ref().unwrap())? {
                self.join_threads().await?;

                info!("Processed tree");
                break;
            }

            let node_to_process = self.find_process_node();

            if let Some(node) = node_to_process {
                trace!("Adding node to process list {}", node.get_id());

                // if self.threads.len() > 5 {
                //     self.join_threads().await?;
                // } else {
                    self.threads
                        .insert(node.get_id(), Box::pin(Gemla::process_node(node)));
                // }
            } else {
                trace!("No node found to process, joining threads");

                self.join_threads().await?;
            }
        }

        Ok(())
    }

    async fn join_threads(&mut self) -> Result<(), Error> {
        if self.threads.len() > 0 {
            trace!("Joining threads for nodes {:?}", self.threads.keys());
    
            let results = futures::future::join_all(self.threads.values_mut()).await;
            let reduced_results: Result<Vec<GeneticNodeWrapper<T>>, Error> =
                results.into_iter().collect();
    
            self.threads.clear();
    
            reduced_results.and_then(|r| {
                if !self
                    .data
                    .mutate(|(d, _)| Gemla::replace_nodes(d.as_mut().unwrap(), r))?
                {
                    warn!("Unable to find nodes to replace in tree")
                }
    
                self.data
                    .mutate(|(d, _)| Gemla::merge_completed_nodes(d.as_mut().unwrap()))??;
    
                Ok(())
            })?;
    
        }

        Ok(())
    }

    fn merge_completed_nodes(tree: &mut SimulationTree<T>) -> Result<(), Error> {
        if tree.val.state() == &GeneticState::Initialize {
            match (&mut tree.left, &mut tree.right) {
                (Some(l), Some(r))
                    if l.val.state() == &GeneticState::Finish
                        && r.val.state() == &GeneticState::Finish =>
                {
                    info!("Merging nodes {} and {}", l.val.get_id(), r.val.get_id());

                    let left_node = l.val.node.as_ref().unwrap();
                    let right_node = r.val.node.as_ref().unwrap();
                    let merged_node = GeneticNode::merge(left_node, right_node)?;
                    tree.val = GeneticNodeWrapper::from(
                        *merged_node,
                        tree.val.total_generations,
                        tree.val.get_id(),
                    );
                }
                (Some(l), Some(r)) => {
                    Gemla::merge_completed_nodes(l)?;
                    Gemla::merge_completed_nodes(r)?;
                }
                (Some(l), None) if l.val.state() == &GeneticState::Finish => {
                    trace!("Copying node {}", l.val.get_id());

                    let left_node = l.val.clone();
                    tree.val = GeneticNodeWrapper::from(
                        left_node.node.unwrap(),
                        tree.val.total_generations,
                        tree.val.get_id(),
                    );
                }
                (Some(l), None) => Gemla::merge_completed_nodes(l)?,
                (None, Some(r)) if r.val.state() == &GeneticState::Finish => {
                    trace!("Copying node {}", r.val.get_id());

                    let right_node = r.val.clone();
                    tree.val = GeneticNodeWrapper::from(
                        right_node.node.unwrap(),
                        tree.val.total_generations,
                        tree.val.get_id(),
                    );
                }
                (None, Some(r)) => Gemla::merge_completed_nodes(r)?,
                (_, _) => (),
            }
        }

        Ok(())
    }

    fn find_process_node_helper(&self, tree: &SimulationTree<T>) -> Option<GeneticNodeWrapper<T>> {
        if tree.val.state() != &GeneticState::Finish
            && !self.threads.contains_key(&tree.val.get_id())
        {
            match (&tree.left, &tree.right) {
                (Some(l), Some(r))
                    if l.val.state() == &GeneticState::Finish
                        && r.val.state() == &GeneticState::Finish =>
                {
                    Some(tree.val.clone())
                }
                (Some(l), Some(r)) => self
                    .find_process_node_helper(&*l)
                    .or_else(|| self.find_process_node_helper(&*r)),
                (Some(l), None) => self.find_process_node_helper(&*l),
                (None, Some(r)) => self.find_process_node_helper(&*r),
                (None, None) => Some(tree.val.clone()),
            }
        } else {
            None
        }
    }

    fn find_process_node(&self) -> Option<GeneticNodeWrapper<T>> {
        let tree = self.data.readonly().0.as_ref();
        tree.and_then(|t| self.find_process_node_helper(&t))
    }

    fn replace_node(
        tree: &mut SimulationTree<T>,
        node: GeneticNodeWrapper<T>,
    ) -> Option<GeneticNodeWrapper<T>> {
        if tree.val.get_id() == node.get_id() {
            tree.val = node;
            None
        } else {
            match (&mut tree.left, &mut tree.right) {
                (Some(l), Some(r)) => {
                    Gemla::replace_node(l, node).and_then(|n| Gemla::replace_node(r, n))
                }
                (Some(l), None) => Gemla::replace_node(l, node),
                (None, Some(r)) => Gemla::replace_node(r, node),
                _ => Some(node),
            }
        }
    }

    fn replace_nodes(tree: &mut SimulationTree<T>, nodes: Vec<GeneticNodeWrapper<T>>) -> bool {
        nodes
            .into_iter()
            .map(|n| Gemla::replace_node(tree, n).is_none())
            .reduce(|a, b| a && b)
            .unwrap_or(false)
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

    async fn process_node(mut node: GeneticNodeWrapper<T>) -> Result<GeneticNodeWrapper<T>, Error> {
        let node_state_time = Instant::now();
        let node_state = *node.state();

        node.process_node()?;

        trace!(
            "{:?} completed in {:?} for {}",
            node_state,
            node_state_time.elapsed(),
            node.get_id()
        );

        if node.state() == &GeneticState::Finish {
            info!("Processed node {}", node.get_id());
        }

        Ok(node)
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
