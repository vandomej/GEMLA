//! Simulates a genetic algorithm on a population in order to improve the fit score and performance. The simulations
//! are performed in a tournament bracket configuration so that populations can compete against each other.

pub mod genetic_node;

use crate::{error::Error, tree::Tree};
use file_linked::{constants::data_format::DataFormat, FileLinked};
use futures::{future, future::BoxFuture};
use genetic_node::{GeneticNode, GeneticNodeWrapper, GeneticState};
use log::{info, trace, warn};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    collections::HashMap, fmt::Debug, fs::File, io::ErrorKind, marker::Send, mem, path::Path,
    time::Instant,
};
use uuid::Uuid;

type SimulationTree<T> = Box<Tree<GeneticNodeWrapper<T>>>;

/// Provides configuration options for managing a [`Gemla`] object as it executes.
/// 
/// # Examples
/// ```rust,ignore
/// #[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
/// struct TestState {
///     pub score: f64,
/// }
/// 
/// impl genetic_node::GeneticNode for TestState {
///     fn simulate(&mut self) -> Result<(), Error> {
///         self.score += 1.0;
///         Ok(())
///     }
/// 
///     fn mutate(&mut self) -> Result<(), Error> {
///         Ok(())
///     }
/// 
///     fn initialize() -> Result<Box<TestState>, Error> {
///         Ok(Box::new(TestState { score: 0.0 }))
///     }
/// 
///     fn merge(left: &TestState, right: &TestState) -> Result<Box<TestState>, Error> {
///         Ok(Box::new(if left.score > right.score {
///             left.clone()
///         } else {
///             right.clone()
///         }))
///     }
/// }
/// 
/// fn main() {
///     
/// }
/// ```
#[derive(Serialize, Deserialize, Copy, Clone)]
pub struct GemlaConfig {
    pub generations_per_height: u64,
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
    threads: HashMap<Uuid, BoxFuture<'a, Result<GeneticNodeWrapper<T>, Error>>>,
}

impl<'a, T: 'a> Gemla<'a, T>
where
    T: GeneticNode + Serialize + DeserializeOwned + Debug + Clone + Send,
{
    pub fn new(path: &Path, config: GemlaConfig, data_format: DataFormat) -> Result<Self, Error> {
        match File::open(path) {
            // If the file exists we either want to overwrite the file or read from the file 
            // based on the configuration provided
            Ok(_) => Ok(Gemla {
                data: if config.overwrite {
                    FileLinked::new((None, config), path, data_format)?
                } else {
                    FileLinked::from_file(path, data_format)?
                },
                threads: HashMap::new(),
            }),
            // If the file doesn't exist we must create it
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(Gemla {
                data: FileLinked::new((None, config), path, data_format)?,
                threads: HashMap::new(),
            }),
            Err(error) => Err(Error::IO(error)),
        }
    }

    pub fn tree_ref(&self) -> Option<&SimulationTree<T>> {
        self.data.readonly().0.as_ref()
    }

    pub async fn simulate(&mut self, steps: u64) -> Result<(), Error> {
        // Only increase height if the tree is uninitialized or completed
        if self.tree_ref().is_none() || 
            self
                .tree_ref()
                .map(|t| Gemla::is_completed(t))
                .unwrap_or(true)
        {
            // Before we can process nodes we must create blank nodes in their place to keep track of which nodes have been processed
            // in the tree and which nodes have not.
            self.data.mutate(|(d, c)| {
                let mut tree: Option<SimulationTree<T>> = Gemla::increase_height(d.take(), c, steps);
                mem::swap(d, &mut tree);
            })?;
        }

        

        info!(
            "Height of simulation tree increased to {}",
            self.tree_ref()
                .map(|t| format!("{}", t.height()))
                .unwrap_or_else(|| "Tree is not defined".to_string())
        );

        loop {
            // We need to keep simulating until the tree has been completely processed.
            if self
                .tree_ref()
                .map(|t| Gemla::is_completed(t))
                .unwrap_or(false)
            {
                self.join_threads().await?;

                info!("Processed tree");
                break;
            }

            if let Some(node) = self
                .tree_ref()
                .and_then(|t| self.get_unprocessed_node(t))
            {
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

            let results = future::join_all(self.threads.values_mut()).await;
            // Converting a list of results into a result wrapping the list
            let reduced_results: Result<Vec<GeneticNodeWrapper<T>>, Error> =
                results.into_iter().collect();
            self.threads.clear();

            // We need to retrieve the processed nodes from the resulting list and replace them in the original list
            reduced_results.and_then(|r| {
                self.data.mutate(|(d, _)| {
                    if let Some(t) = d {
                        let failed_nodes = Gemla::replace_nodes(t, r);
                        // We receive a list of nodes that were unable to be found in the original tree
                        if !failed_nodes.is_empty() {
                            warn!(
                                "Unable to find {:?} to replace in tree",
                                failed_nodes.iter().map(|n| n.id())
                            )
                        }

                        // Once the nodes are replaced we need to find nodes that can be merged from the completed children nodes
                        Gemla::merge_completed_nodes(t)
                    } else {
                        warn!("Unable to replce nodes {:?} in empty tree", r);
                        Ok(())
                    }
                })?
            })?;
        }

        Ok(())
    }

    fn merge_completed_nodes(tree: &mut SimulationTree<T>) -> Result<(), Error> {
        if tree.val.state() == GeneticState::Initialize {
            match (&mut tree.left, &mut tree.right) {
                // If the current node has been initialized, and has children nodes that are completed, then we need
                // to merge the children nodes together into the parent node
                (Some(l), Some(r))
                    if l.val.state() == GeneticState::Finish
                        && r.val.state() == GeneticState::Finish =>
                {
                    info!("Merging nodes {} and {}", l.val.id(), r.val.id());
                    if let (Some(left_node), Some(right_node)) = (l.val.as_ref(), r.val.as_ref()) {
                        let merged_node = GeneticNode::merge(left_node, right_node, &tree.val.id())?;
                        tree.val = GeneticNodeWrapper::from(
                            *merged_node,
                            tree.val.max_generations(),
                            tree.val.id(),
                        );
                    }
                }
                (Some(l), Some(r)) => {
                    Gemla::merge_completed_nodes(l)?;
                    Gemla::merge_completed_nodes(r)?;
                }
                // If there is only one child node that's completed then we want to copy it to the parent node
                (Some(l), None) if l.val.state() == GeneticState::Finish => {
                    trace!("Copying node {}", l.val.id());

                    if let Some(left_node) = l.val.as_ref() {
                        GeneticNodeWrapper::from(
                            left_node.clone(),
                            tree.val.max_generations(),
                            tree.val.id(),
                        );
                    }
                }
                (Some(l), None) => Gemla::merge_completed_nodes(l)?,
                (None, Some(r)) if r.val.state() == GeneticState::Finish => {
                    trace!("Copying node {}", r.val.id());

                    if let Some(right_node) = r.val.as_ref() {
                        tree.val = GeneticNodeWrapper::from(
                            right_node.clone(),
                            tree.val.max_generations(),
                            tree.val.id(),
                        );
                    }
                }
                (None, Some(r)) => Gemla::merge_completed_nodes(r)?,
                (_, _) => (),
            }
        }

        Ok(())
    }

    fn get_unprocessed_node(&self, tree: &SimulationTree<T>) -> Option<GeneticNodeWrapper<T>> {
        // If the current node has been processed or exists in the thread list then we want to stop recursing. Checking if it exists in the thread list 
        // should be fine because we process the tree from bottom to top.
        if tree.val.state() != GeneticState::Finish && !self.threads.contains_key(&tree.val.id()) {
            match (&tree.left, &tree.right) {
                // If the children are finished we can start processing the currrent node. The current node should be merged from the children already 
                // during join_threads.
                (Some(l), Some(r))
                    if l.val.state() == GeneticState::Finish
                        && r.val.state() == GeneticState::Finish => Some(tree.val.clone()),
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
        // Replacing nodes as we recurse through the tree
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
            let left_branch_height =
                tree.as_ref().map(|t| t.height() as u64).unwrap_or(0) + amount - 1;
            
            Some(Box::new(Tree::new(
                GeneticNodeWrapper::new(config.generations_per_height),
                Gemla::increase_height(tree, config, amount - 1),
                // The right branch height has to equal the left branches total height
                if left_branch_height > 0 {
                    Some(Box::new(btree!(GeneticNodeWrapper::new(
                        left_branch_height * config.generations_per_height
                    ))))
                } else {
                    None
                },
            )))
        }
    }

    fn is_completed(tree: &SimulationTree<T>) -> bool {
        // If the current node is finished, then by convention the children should all be finished as well
        tree.val.state() == GeneticState::Finish 
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
    use std::path::PathBuf;
    use std::fs;

    use self::genetic_node::GeneticNodeContext;

    struct CleanUp {
        path: PathBuf,
    }

    impl CleanUp {
        fn new(path: &Path) -> CleanUp {
            CleanUp {
                path: path.to_path_buf(),
            }
        }

        pub fn run<F: FnOnce(&Path) -> Result<(), Error>>(&self, op: F) -> Result<(), Error> {
            op(&self.path)
        }
    }

    impl Drop for CleanUp {
        fn drop(&mut self) {
            if self.path.exists() {
                fs::remove_file(&self.path).expect("Unable to remove file");
            }
        }
    }

    #[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
    struct TestState {
        pub score: f64,
    }

    impl genetic_node::GeneticNode for TestState {
        fn simulate(&mut self, _context: &GeneticNodeContext) -> Result<(), Error> {
            self.score += 1.0;
            Ok(())
        }

        fn mutate(&mut self, _context: &GeneticNodeContext) -> Result<(), Error> {
            Ok(())
        }

        fn initialize(_context: &GeneticNodeContext) -> Result<Box<TestState>, Error> {
            Ok(Box::new(TestState { score: 0.0 }))
        }

        fn merge(left: &TestState, right: &TestState, _id: &Uuid) -> Result<Box<TestState>, Error> {
            Ok(Box::new(if left.score > right.score {
                left.clone()
            } else {
                right.clone()
            }))
        }
    }

    #[test]
    fn test_new() -> Result<(), Error> {
        let path = PathBuf::from("test_new_non_existing");
        CleanUp::new(&path).run(|p| {
            assert!(!path.exists());

            // Testing initial creation
            let mut config = GemlaConfig {
                generations_per_height: 1,
                overwrite: true
            };
            let mut gemla = Gemla::<TestState>::new(&p, config, DataFormat::Json)?;

            smol::block_on(gemla.simulate(2))?;
            assert_eq!(gemla.data.readonly().0.as_ref().unwrap().height(), 2);
            
            drop(gemla);
            assert!(path.exists());

            // Testing overwriting data
            let mut gemla = Gemla::<TestState>::new(&p, config, DataFormat::Json)?;

            smol::block_on(gemla.simulate(2))?;
            assert_eq!(gemla.data.readonly().0.as_ref().unwrap().height(), 2);

            drop(gemla);
            assert!(path.exists());

            // Testing not-overwriting data
            config.overwrite = false;
            let mut gemla = Gemla::<TestState>::new(&p, config, DataFormat::Json)?;

            smol::block_on(gemla.simulate(2))?;
            assert_eq!(gemla.tree_ref().unwrap().height(), 4);

            drop(gemla);
            assert!(path.exists());

            Ok(())
        })
    }

    #[test]
    fn test_simulate() -> Result<(), Error> {
        let path = PathBuf::from("test_simulate");
        CleanUp::new(&path).run(|p| {
            // Testing initial creation
            let config = GemlaConfig {
                generations_per_height: 10,
                overwrite: true
            };
            let mut gemla = Gemla::<TestState>::new(&p, config, DataFormat::Json)?;

            smol::block_on(gemla.simulate(5))?;
            let tree = gemla.tree_ref().unwrap();
            assert_eq!(tree.height(), 5);
            assert_eq!(tree.val.as_ref().unwrap().score, 50.0);

            Ok(())
        })
    }

}
