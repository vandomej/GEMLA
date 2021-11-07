//! Simulates a genetic algorithm on a population in order to improve the fit score and performance. The simulations
//! are performed in a tournament bracket configuration so that populations can compete against each other.

pub mod genetic_node;

use crate::error::Error;
use crate::tree::Tree;
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

type SimulationTree<T> = Box<Tree<GeneticNodeWrapper<T>>>;

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
            Ok(_) => Ok(Gemla {
                data: if config.overwrite {
                    FileLinked::new((None, config), path)?
                } else {
                    FileLinked::from_file(path)?
                },
                threads: std::collections::HashMap::new(),
            }),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(Gemla {
                data: FileLinked::new((None, config), path)?,
                threads: std::collections::HashMap::new(),
            }),
            Err(error) => Err(Error::IO(error)),
        }
    }

    pub async fn simulate(&mut self, steps: u64) -> Result<(), Error> {
        // Before we can process nodes we must create blank nodes in their place to keep track of which nodes have been processed
        // in the tree and which nodes have not.
        self.data.mutate(|(d, c)| {
            let mut tree: Option<SimulationTree<T>> = Gemla::increase_height(d.take(), c, steps);
            swap(d, &mut tree);
        })?;

        // println!("{}", serde_json::to_string(&self.data.readonly().0).expect(""));

        info!(
            "Height of simulation tree increased to {}",
            self.data.readonly().0.as_ref().unwrap().height()
        );

        loop {
            if Gemla::is_completed(self.data.readonly().0.as_ref().unwrap()) {
                self.join_threads().await?;

                info!("Processed tree");
                break;
            }

            let node_to_process =
                self.get_unprocessed_node(self.data.readonly().0.as_ref().unwrap());

            if let Some(node) = node_to_process {
                trace!("Adding node to process list {}", node.id());

                self.threads
                    .insert(node.id(), Box::pin(Gemla::process_node(node)));
            } else {
                trace!("No node found to process, joining threads");

                self.join_threads().await?;
            }
        }

        Ok(())
    }

    async fn join_threads(&mut self) -> Result<(), Error> {
        if !self.threads.is_empty() {
            trace!("Joining threads for nodes {:?}", self.threads.keys());

            let results = futures::future::join_all(self.threads.values_mut()).await;
            let reduced_results: Result<Vec<GeneticNodeWrapper<T>>, Error> =
                results.into_iter().collect();

            self.threads.clear();

            reduced_results.and_then(|r| {
                let failed_nodes = self
                    .data
                    .mutate(|(d, _)| Gemla::replace_nodes(d.as_mut().unwrap(), r))?;

                if !failed_nodes.is_empty() {
                    warn!(
                        "Unable to find {:?} to replace in tree",
                        failed_nodes.iter().map(|n| n.id())
                    )
                }

                self.data
                    .mutate(|(d, _)| Gemla::merge_completed_nodes(d.as_mut().unwrap()))??;

                Ok(())
            })?;
        }

        Ok(())
    }

    fn merge_completed_nodes(tree: &mut SimulationTree<T>) -> Result<(), Error> {
        if tree.val.state() == GeneticState::Initialize {
            match (&mut tree.left, &mut tree.right) {
                (Some(l), Some(r))
                    if l.val.state() == GeneticState::Finish
                        && r.val.state() == GeneticState::Finish =>
                {
                    info!("Merging nodes {} and {}", l.val.id(), r.val.id());

                    let left_node = l.val.as_ref().unwrap();
                    let right_node = r.val.as_ref().unwrap();
                    let merged_node = GeneticNode::merge(left_node, right_node)?;
                    tree.val = GeneticNodeWrapper::from(
                        *merged_node,
                        tree.val.max_generations(),
                        tree.val.id(),
                    );
                }
                (Some(l), Some(r)) => {
                    Gemla::merge_completed_nodes(l)?;
                    Gemla::merge_completed_nodes(r)?;
                }
                (Some(l), None) if l.val.state() == GeneticState::Finish => {
                    trace!("Copying node {}", l.val.id());

                    tree.val = GeneticNodeWrapper::from(
                        l.val.as_ref().unwrap().clone(),
                        tree.val.max_generations(),
                        tree.val.id(),
                    );
                }
                (Some(l), None) => Gemla::merge_completed_nodes(l)?,
                (None, Some(r)) if r.val.state() == GeneticState::Finish => {
                    trace!("Copying node {}", r.val.id());

                    tree.val = GeneticNodeWrapper::from(
                        r.val.as_ref().unwrap().clone(),
                        tree.val.max_generations(),
                        tree.val.id(),
                    );
                }
                (None, Some(r)) => Gemla::merge_completed_nodes(r)?,
                (_, _) => (),
            }
        }

        Ok(())
    }

    fn get_unprocessed_node(&self, tree: &SimulationTree<T>) -> Option<GeneticNodeWrapper<T>> {
        if tree.val.state() != GeneticState::Finish && !self.threads.contains_key(&tree.val.id()) {
            match (&tree.left, &tree.right) {
                (Some(l), Some(r))
                    if l.val.state() == GeneticState::Finish
                        && r.val.state() == GeneticState::Finish =>
                {
                    Some(tree.val.clone())
                }
                (Some(l), Some(r)) => self
                    .get_unprocessed_node(l)
                    .or_else(|| self.get_unprocessed_node(r)),
                (Some(l), None) => self.get_unprocessed_node(l),
                (None, Some(r)) => self.get_unprocessed_node(r),
                (None, None) => Some(tree.val.clone()),
            }
        } else {
            None
        }
    }

    fn replace_nodes(
        tree: &mut SimulationTree<T>,
        mut nodes: Vec<GeneticNodeWrapper<T>>,
    ) -> Vec<GeneticNodeWrapper<T>> {
        if let Some(i) = nodes.iter().position(|n| n.id() == tree.val.id()) {
            tree.val = nodes.remove(i);
        }

        match (&mut tree.left, &mut tree.right) {
            (Some(l), Some(r)) => Gemla::replace_nodes(r, Gemla::replace_nodes(l, nodes)),
            (Some(l), None) => Gemla::replace_nodes(l, nodes),
            (None, Some(r)) => Gemla::replace_nodes(r, nodes),
            _ => nodes,
        }
    }

    fn increase_height(
        tree: Option<SimulationTree<T>>,
        config: &GemlaConfig,
        amount: u64,
    ) -> Option<SimulationTree<T>> {
        if amount == 0 {
            tree
        } else {
            let right_branch_height =
                tree.as_ref().map(|t| t.height() as u64).unwrap_or(0) + amount - 1;

            Some(Box::new(Tree::new(
                GeneticNodeWrapper::new(config.generations_per_node),
                Gemla::increase_height(tree, config, amount - 1),
                if right_branch_height > 0 {
                    Some(Box::new(btree!(GeneticNodeWrapper::new(
                        right_branch_height * config.generations_per_node
                    ))))
                } else {
                    None
                },
            )))
        }
    }

    fn is_completed(tree: &SimulationTree<T>) -> bool {
        if tree.val.state() == GeneticState::Finish {
            match (&tree.left, &tree.right) {
                (Some(l), Some(r)) => Gemla::is_completed(l) && Gemla::is_completed(r),
                (Some(l), None) => Gemla::is_completed(l),
                (None, Some(r)) => Gemla::is_completed(r),
                (None, None) => true,
            }
        } else {
            false
        }
    }

    async fn process_node(mut node: GeneticNodeWrapper<T>) -> Result<GeneticNodeWrapper<T>, Error> {
        let node_state_time = Instant::now();
        let node_state = node.state();

        node.process_node()?;

        trace!(
            "{:?} completed in {:?} for {}",
            node_state,
            node_state_time.elapsed(),
            node.id()
        );

        if node.state() == GeneticState::Finish {
            info!("Processed node {}", node.id());
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
