//! Simulates a genetic algorithm on a population in order to improve the fit score and performance. The simulations
//! are performed in a tournament bracket configuration so that populations can compete against each other.

pub mod genetic_node;

use crate::{error::Error, tree::Tree};
use async_recursion::async_recursion;
use file_linked::{constants::data_format::DataFormat, FileLinked};
use futures::future;
use genetic_node::{GeneticNode, GeneticNodeWrapper, GeneticState};
use log::{info, trace, warn};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    collections::HashMap, fmt::Debug, fs::File, io::ErrorKind, marker::Send, mem, path::Path,
    sync::Arc, time::Instant,
};
use tokio::{sync::RwLock, task::JoinHandle};
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
pub struct Gemla<T>
where
    T: GeneticNode + Serialize + DeserializeOwned + Debug + Send + Clone,
    T::Context: Send + Sync + Clone + Debug + Serialize + DeserializeOwned + 'static + Default,
{
    pub data: FileLinked<(Option<SimulationTree<T>>, GemlaConfig, T::Context)>,
    threads: HashMap<Uuid, JoinHandle<Result<GeneticNodeWrapper<T>, Error>>>,
}

impl<T: 'static> Gemla<T>
where
    T: GeneticNode + Serialize + DeserializeOwned + Debug + Send + Sync + Clone,
    T::Context: Send + Sync + Clone + Debug + Serialize + DeserializeOwned + 'static + Default,
{
    pub async fn new(
        path: &Path,
        config: GemlaConfig,
        data_format: DataFormat,
    ) -> Result<Self, Error> {
        match File::open(path) {
            // If the file exists we either want to overwrite the file or read from the file
            // based on the configuration provided
            Ok(_) => Ok(Gemla {
                data: if config.overwrite {
                    FileLinked::new((None, config, T::Context::default()), path, data_format)
                        .await?
                } else {
                    FileLinked::from_file(path, data_format)?
                },
                threads: HashMap::new(),
            }),
            // If the file doesn't exist we must create it
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(Gemla {
                data: FileLinked::new((None, config, T::Context::default()), path, data_format)
                    .await?,
                threads: HashMap::new(),
            }),
            Err(error) => Err(Error::IO(error)),
        }
    }

    pub fn tree_ref(&self) -> Arc<RwLock<(Option<SimulationTree<T>>, GemlaConfig, T::Context)>> {
        self.data.readonly().clone()
    }

    pub async fn simulate(&mut self, steps: u64) -> Result<(), Error> {
        let tree_completed = {
            // Only increase height if the tree is uninitialized or completed
            let data_arc = self.data.readonly();
            let data_ref = data_arc.read().await;
            let tree_ref = data_ref.0.as_ref();

            tree_ref.is_none() || tree_ref.map(|t| Gemla::is_completed(t)).unwrap_or(true)
        };

        if tree_completed {
            // Before we can process nodes we must create blank nodes in their place to keep track of which nodes have been processed
            // in the tree and which nodes have not.
            self.data
                .mutate(|(d, c, _)| {
                    let mut tree: Option<SimulationTree<T>> =
                        Gemla::increase_height(d.take(), c, steps);
                    mem::swap(d, &mut tree);
                })
                .await?;
        }

        {
            // Only increase height if the tree is uninitialized or completed
            let data_arc = self.data.readonly();
            let data_ref = data_arc.read().await;
            let tree_ref = data_ref.0.as_ref();

            info!(
                "Height of simulation tree increased to {}",
                tree_ref
                    .map(|t| format!("{}", t.height()))
                    .unwrap_or_else(|| "Tree is not defined".to_string())
            );
        }

        loop {
            let is_tree_processed;

            {
                let data_arc = self.data.readonly();
                let data_ref = data_arc.read().await;
                let tree_ref = data_ref.0.as_ref();

                is_tree_processed = tree_ref.map(|t| Gemla::is_completed(t)).unwrap_or(false)
            }

            // We need to keep simulating until the tree has been completely processed.
            if is_tree_processed {
                self.join_threads().await?;

                info!("Processed tree");
                break;
            }

            let (node, gemla_context) = {
                let data_arc = self.data.readonly();
                let data_ref = data_arc.read().await;
                let (tree_ref, _, gemla_context) = &*data_ref; // (Option<Box<Tree<GeneticNodeWrapper<T>>>, GemlaConfig, T::Context)

                let node = tree_ref.as_ref().and_then(|t| self.get_unprocessed_node(t));

                (node, gemla_context.clone())
            };

            if let Some(node) = node {
                trace!("Adding node to process list {}", node.id());

                let gemla_context = gemla_context.clone();

                self.threads.insert(
                    node.id(),
                    tokio::spawn(async move { Gemla::process_node(node, gemla_context).await }),
                );
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
                results.into_iter().flatten().collect();
            self.threads.clear();

            // We need to retrieve the processed nodes from the resulting list and replace them in the original list
            match reduced_results {
                Ok(r) => {
                    self.data
                        .mutate_async(|d| async move {
                            // Scope to limit the duration of the read lock
                            let (_, context) = {
                                let data_read = d.read().await;
                                (data_read.1, data_read.2.clone())
                            }; // Read lock is dropped here

                            let mut data_write = d.write().await;

                            if let Some(t) = data_write.0.as_mut() {
                                let failed_nodes = Gemla::replace_nodes(t, r);
                                // We receive a list of nodes that were unable to be found in the original tree
                                if !failed_nodes.is_empty() {
                                    warn!(
                                        "Unable to find {:?} to replace in tree",
                                        failed_nodes.iter().map(|n| n.id())
                                    )
                                }

                                // Once the nodes are replaced we need to find nodes that can be merged from the completed children nodes
                                Gemla::merge_completed_nodes(t, context.clone()).await
                            } else {
                                warn!("Unable to replce nodes {:?} in empty tree", r);
                                Ok(())
                            }
                        })
                        .await??;
                }
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    #[async_recursion]
    async fn merge_completed_nodes<'a>(
        tree: &'a mut SimulationTree<T>,
        gemla_context: T::Context,
    ) -> Result<(), Error> {
        if tree.val.state() == GeneticState::Initialize {
            match (&mut tree.left, &mut tree.right) {
                // If the current node has been initialized, and has children nodes that are completed, then we need
                // to merge the children nodes together into the parent node
                (Some(l), Some(r))
                    if l.val.state() == GeneticState::Finish
                        && r.val.state() == GeneticState::Finish =>
                {
                    info!("Merging nodes {} and {}", l.val.id(), r.val.id());
                    if let (Some(left_node), Some(right_node)) = (l.val.take(), r.val.take()) {
                        let merged_node = GeneticNode::merge(
                            &left_node,
                            &right_node,
                            &tree.val.id(),
                            gemla_context.clone(),
                        )
                        .await?;
                        tree.val = GeneticNodeWrapper::from(
                            *merged_node,
                            tree.val.max_generations(),
                            tree.val.id(),
                        );
                    }
                }
                (Some(l), Some(r)) => {
                    Gemla::merge_completed_nodes(l, gemla_context.clone()).await?;
                    Gemla::merge_completed_nodes(r, gemla_context.clone()).await?;
                }
                // If there is only one child node that's completed then we want to copy it to the parent node
                (Some(l), None) if l.val.state() == GeneticState::Finish => {
                    trace!("Copying node {}", l.val.id());

                    if let Some(left_node) = l.val.take() {
                        GeneticNodeWrapper::from(
                            left_node,
                            tree.val.max_generations(),
                            tree.val.id(),
                        );
                    }
                }
                (Some(l), None) => Gemla::merge_completed_nodes(l, gemla_context.clone()).await?,
                (None, Some(r)) if r.val.state() == GeneticState::Finish => {
                    trace!("Copying node {}", r.val.id());

                    if let Some(right_node) = r.val.take() {
                        tree.val = GeneticNodeWrapper::from(
                            right_node,
                            tree.val.max_generations(),
                            tree.val.id(),
                        );
                    }
                }
                (None, Some(r)) => Gemla::merge_completed_nodes(r, gemla_context.clone()).await?,
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

    async fn process_node(
        mut node: GeneticNodeWrapper<T>,
        gemla_context: T::Context,
    ) -> Result<GeneticNodeWrapper<T>, Error> {
        let node_state_time = Instant::now();
        let node_state = node.state();

        node.process_node(gemla_context.clone()).await?;

        info!(
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
    use async_trait::async_trait;
    use serde::{Deserialize, Serialize};
    use std::fs;
    use std::path::PathBuf;
    use tokio::runtime::Runtime;

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

    #[async_trait]
    impl genetic_node::GeneticNode for TestState {
        type Context = ();

        async fn simulate(
            &mut self,
            _context: GeneticNodeContext<Self::Context>,
        ) -> Result<(), Error> {
            self.score += 1.0;
            Ok(())
        }

        async fn mutate(
            &mut self,
            _context: GeneticNodeContext<Self::Context>,
        ) -> Result<(), Error> {
            Ok(())
        }

        async fn initialize(
            _context: GeneticNodeContext<Self::Context>,
        ) -> Result<Box<TestState>, Error> {
            Ok(Box::new(TestState { score: 0.0 }))
        }

        async fn merge(
            left: &TestState,
            right: &TestState,
            _id: &Uuid,
            _: Self::Context,
        ) -> Result<Box<TestState>, Error> {
            Ok(Box::new(if left.score > right.score {
                left.clone()
            } else {
                right.clone()
            }))
        }
    }

    #[tokio::test]
    async fn test_new() -> Result<(), Error> {
        let path = PathBuf::from("test_new_non_existing");
        // Use `spawn_blocking` to run synchronous code that needs to call async code internally.
        tokio::task::spawn_blocking(move || {
            let rt = Runtime::new().unwrap(); // Create a new Tokio runtime for the async block.
            CleanUp::new(&path).run(move |p| {
                rt.block_on(async {
                    assert!(!path.exists());

                    // Testing initial creation
                    let mut config = GemlaConfig {
                        generations_per_height: 1,
                        overwrite: true,
                    };
                    let mut gemla = Gemla::<TestState>::new(&p, config, DataFormat::Json).await?;

                    // Now we can use `.await` within the spawned blocking task.
                    gemla.simulate(2).await?;
                    let data = gemla.data.readonly();
                    let data_lock = data.read().await;
                    assert_eq!(data_lock.0.as_ref().unwrap().height(), 2);

                    drop(data_lock);
                    drop(gemla);
                    assert!(path.exists());

                    // Testing overwriting data
                    let mut gemla = Gemla::<TestState>::new(&p, config, DataFormat::Json).await?;

                    gemla.simulate(2).await?;
                    let data = gemla.data.readonly();
                    let data_lock = data.read().await;
                    assert_eq!(data_lock.0.as_ref().unwrap().height(), 2);

                    drop(data_lock);
                    drop(gemla);
                    assert!(path.exists());

                    // Testing not-overwriting data
                    config.overwrite = false;
                    let mut gemla = Gemla::<TestState>::new(&p, config, DataFormat::Json).await?;

                    gemla.simulate(2).await?;
                    let data = gemla.data.readonly();
                    let data_lock = data.read().await;
                    let tree = data_lock.0.as_ref().unwrap();
                    assert_eq!(tree.height(), 4);

                    drop(data_lock);
                    drop(gemla);
                    assert!(path.exists());

                    Ok(())
                })
            })
        })
        .await
        .unwrap()?; // Wait for the blocking task to complete, then handle the Result.

        Ok(())
    }

    #[tokio::test]
    async fn test_simulate() -> Result<(), Error> {
        let path = PathBuf::from("test_simulate");
        // Use `spawn_blocking` to run the synchronous closure that internally awaits async code.
        tokio::task::spawn_blocking(move || {
            let rt = Runtime::new().unwrap(); // Create a new Tokio runtime for the async block.
            CleanUp::new(&path).run(move |p| {
                rt.block_on(async {
                    // Testing initial creation
                    let config = GemlaConfig {
                        generations_per_height: 10,
                        overwrite: true,
                    };
                    let mut gemla = Gemla::<TestState>::new(&p, config, DataFormat::Json).await?;

                    // Now we can use `.await` within the spawned blocking task.
                    gemla.simulate(5).await?;
                    let data = gemla.data.readonly();
                    let data_lock = data.read().await;
                    let tree = data_lock.0.as_ref().unwrap();
                    assert_eq!(tree.height(), 5);
                    assert_eq!(tree.val.as_ref().unwrap().score, 50.0);

                    Ok(())
                })
            })
        })
        .await
        .unwrap()?; // Wait for the blocking task to complete, then handle the Result.

        Ok(())
    }
}
