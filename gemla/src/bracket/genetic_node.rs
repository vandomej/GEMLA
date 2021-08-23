//! A trait used to interact with the internal state of nodes within the [`Bracket`]
//!
//! [`Bracket`]: crate::bracket::Bracket

use serde::{Deserialize, Serialize};
use std::fmt;

/// An enum used to control the state of a [`GeneticNode`]
///
/// [`GeneticNode`]: crate::bracket::genetic_node
#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
#[serde(tag = "enumType", content = "enumContent")]
pub enum GeneticState {
    /// The node and it's data have not finished initializing
    Initialize,
    /// The node is currently simulating a round against target data to determine the fitness of the population
    Simulate,
    /// The node is currently selecting members of the population that scored well and reducing the total population size
    Score,
    /// The node is currently mutating members of it's population and breeding new members
    Mutate,
    /// The node has finished processing for a given number of iterations
    Finish,
}

/// A trait used to interact with the internal state of nodes within the [`Bracket`]
///
/// [`Bracket`]: crate::bracket::Bracket
pub trait GeneticNode {
    /// Initializes a new instance of a [`GeneticState`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use gemla::bracket::genetic_node::GeneticNode;
    /// #
    /// struct Node {
    ///     pub fit_score: f64,
    /// }
    ///
    /// impl GeneticNode for Node {
    ///     fn initialize() -> Result<Box<Self>, String> {
    ///         Ok(Box::new(Node {fit_score: 0.0}))
    ///     }
    ///
    ///     //...
    /// #
    /// #   fn simulate(&mut self, iterations: u64) -> Result<(), String> {
    /// #       Ok(())
    /// #   }
    /// #
    /// #   fn get_fit_score(&self) -> f64 {
    /// #       self.fit_score
    /// #   }
    /// #
    /// #   fn calculate_scores_and_trim(&mut self) -> Result<(), String> {
    /// #       Ok(())
    /// #   }
    /// #
    /// #   fn mutate(&mut self) -> Result<(), String> {
    /// #       Ok(())
    /// #   }
    /// }
    ///
    /// # fn main() -> Result<(), String> {
    /// let node = Node::initialize()?;
    /// assert_eq!(node.get_fit_score(), 0.0);
    /// # Ok(())
    /// # }
    /// ```
    fn initialize() -> Result<Box<Self>, String>;

    /// Runs a simulation on the state object for the given number of `iterations` in order to guage it's fitness.
    /// This will be called for every node in a bracket before evaluating it's fitness against other nodes.
    ///
    /// #Examples
    ///
    /// ```
    /// # use gemla::bracket::genetic_node::GeneticNode;
    /// #
    /// struct Model {
    ///     pub fit_score: f64,
    ///     //...
    /// }
    ///
    /// struct Node {
    ///     pub models: Vec<Model>,
    ///     //...
    /// }
    ///
    /// impl Model {
    ///     fn fit(&mut self, epochs: u64) -> Result<(), String> {
    ///         //...
    /// #        self.fit_score += epochs as f64;
    /// #        Ok(())
    ///     }
    /// }
    ///    
    /// impl GeneticNode for Node {
    /// #    fn initialize() -> Result<Box<Self>, String> {
    /// #        Ok(Box::new(Node {models: vec![Model {fit_score: 0.0}]}))
    /// #    }
    /// #
    ///     //...
    ///
    ///     fn simulate(&mut self, iterations: u64) -> Result<(), String> {
    ///            for m in self.models.iter_mut()
    ///            {
    ///                m.fit(iterations)?;
    ///            }
    ///            Ok(())
    ///        }
    ///
    ///     //...
    ///
    /// #    fn get_fit_score(&self) -> f64 {
    /// #        self.models.iter().max_by(|m1, m2| m1.fit_score.partial_cmp(&m2.fit_score).unwrap()).unwrap().fit_score
    /// #    }
    /// #   
    /// #    fn calculate_scores_and_trim(&mut self) -> Result<(), String> {
    /// #        Ok(())
    /// #    }
    /// #   
    /// #    fn mutate(&mut self) -> Result<(), String> {
    /// #        Ok(())
    /// #    }
    /// }
    ///    
    /// # fn main() -> Result<(), String> {
    /// let mut node = Node::initialize()?;
    /// node.simulate(5)?;
    /// assert_eq!(node.get_fit_score(), 5.0);
    /// #    Ok(())
    /// # }
    /// ```
    fn simulate(&mut self, iterations: u64) -> Result<(), String>;

    /// Returns a fit score associated with the nodes performance.
    /// This will be used by a bracket in order to determine the most successful child.
    ///
    /// # Examples
    /// ```
    /// # use gemla::bracket::genetic_node::GeneticNode;
    /// #
    /// struct Model {
    ///     pub fit_score: f64,
    ///     //...
    /// }
    ///
    /// struct Node {
    ///     pub models: Vec<Model>,
    ///     //...
    /// }
    ///
    /// # impl Model {
    /// #     fn fit(&mut self, epochs: u64) -> Result<(), String> {
    /// #         //...
    /// #        self.fit_score += epochs as f64;
    /// #        Ok(())
    /// #     }
    /// # }
    ///    
    /// impl GeneticNode for Node {
    /// #    fn initialize() -> Result<Box<Self>, String> {
    /// #        Ok(Box::new(Node {models: vec![Model {fit_score: 0.0}]}))
    /// #    }
    /// #
    /// #     //...
    /// #
    /// #    fn simulate(&mut self, iterations: u64) -> Result<(), String> {
    /// #           for m in self.models.iter_mut()
    /// #           {
    /// #               m.fit(iterations)?;
    /// #           }
    /// #           Ok(())
    /// #       }
    /// #
    ///     //...
    ///
    ///     fn get_fit_score(&self) -> f64 {
    ///         self.models.iter().max_by(|m1, m2| m1.fit_score.partial_cmp(&m2.fit_score).unwrap()).unwrap().fit_score
    ///     }
    ///
    ///     //...   
    /// #    fn calculate_scores_and_trim(&mut self) -> Result<(), String> {
    /// #        Ok(())
    /// #    }
    /// #   
    /// #    fn mutate(&mut self) -> Result<(), String> {
    /// #        Ok(())
    /// #    }
    /// }
    ///    
    /// # fn main() -> Result<(), String> {
    /// let mut node = Node::initialize()?;
    /// node.simulate(5)?;
    /// assert_eq!(node.get_fit_score(), 5.0);
    /// #    Ok(())
    /// # }
    /// ```
    fn get_fit_score(&self) -> f64;

    /// Used when scoring the nodes after simulating and should remove underperforming children.
    ///
    /// # Examples
    /// ```
    /// # use gemla::bracket::genetic_node::GeneticNode;
    /// #
    /// struct Model {
    ///     pub fit_score: f64,
    ///     //...
    /// }
    ///
    /// struct Node {
    ///     pub models: Vec<Model>,
    ///     population_size: i64,
    ///     //...
    /// }
    ///
    /// # impl Model {
    /// #     fn fit(&mut self, epochs: u64) -> Result<(), String> {
    /// #         //...
    /// #        self.fit_score += epochs as f64;
    /// #         Ok(())
    /// #     }
    /// # }
    ///
    /// impl GeneticNode for Node {
    /// #     fn initialize() -> Result<Box<Self>, String> {
    /// #         Ok(Box::new(Node {
    /// #             models: vec![
    /// #                 Model { fit_score: 0.0 },
    /// #                 Model { fit_score: 1.0 },
    /// #                 Model { fit_score: 2.0 },
    /// #                 Model { fit_score: 3.0 },
    /// #                 Model { fit_score: 4.0 },
    /// #             ],
    /// #             population_size: 5,
    /// #         }))
    /// #     }
    /// #
    /// #    //...
    /// #
    /// #     fn simulate(&mut self, iterations: u64) -> Result<(), String> {
    /// #         for m in self.models.iter_mut() {
    /// #             m.fit(iterations)?;
    /// #         }
    /// #         Ok(())
    /// #     }
    /// #
    ///     //...
    ///
    /// #    fn get_fit_score(&self) -> f64 {
    /// #        self.models
    /// #            .iter()
    /// #            .max_by(|m1, m2| m1.fit_score.partial_cmp(&m2.fit_score).unwrap())
    /// #            .unwrap()
    /// #            .fit_score
    /// #    }
    /// #
    ///     fn calculate_scores_and_trim(&mut self) -> Result<(), String> {
    ///         self.models.sort_by(|a, b| a.fit_score.partial_cmp(&b.fit_score).unwrap().reverse());
    ///         self.models.truncate(3);
    ///         Ok(())
    ///     }
    ///
    ///     //...
    /// #
    /// #     fn mutate(&mut self) -> Result<(), String> {
    /// #         Ok(())
    /// #     }
    /// }
    ///
    /// # fn main() -> Result<(), String> {
    /// let mut node = Node::initialize()?;
    /// assert_eq!(node.models.len(), 5);
    ///
    /// node.simulate(5)?;
    /// node.calculate_scores_and_trim()?;
    /// assert_eq!(node.models.len(), 3);
    ///
    /// # assert_eq!(node.get_fit_score(), 9.0);
    /// # Ok(())
    /// # }
    /// ```
    fn calculate_scores_and_trim(&mut self) -> Result<(), String>;

    /// Mutates members in a population and/or crossbreeds them to produce new offspring.
    ///
    /// # Examples
    /// ```
    /// # use gemla::bracket::genetic_node::GeneticNode;
    /// # use std::convert::TryInto;
    /// #
    /// struct Model {
    ///     pub fit_score: f64,
    ///     //...
    /// }
    ///
    /// struct Node {
    ///     pub models: Vec<Model>,
    ///     population_size: i64,
    ///     //...
    /// }
    ///
    /// # impl Model {
    /// #     fn fit(&mut self, epochs: u64) -> Result<(), String> {
    /// #         //...
    /// #         self.fit_score += epochs as f64;
    /// #         Ok(())
    /// #     }
    /// # }
    ///
    /// fn mutate_random_individuals(_models: &Vec<Model>) -> Model
    /// {
    ///     //...
    /// #     Model {
    /// #         fit_score: 0.0
    /// #     }
    /// }
    ///
    /// impl GeneticNode for Node {
    /// #     fn initialize() -> Result<Box<Self>, String> {
    /// #         Ok(Box::new(Node {
    /// #             models: vec![
    /// #                 Model { fit_score: 0.0 },
    /// #                 Model { fit_score: 1.0 },
    /// #                 Model { fit_score: 2.0 },
    /// #                 Model { fit_score: 3.0 },
    /// #                 Model { fit_score: 4.0 },
    /// #             ],
    /// #             population_size: 5,
    /// #         }))
    /// #     }
    /// #
    /// #     fn simulate(&mut self, iterations: u64) -> Result<(), String> {
    /// #         for m in self.models.iter_mut() {
    /// #             m.fit(iterations)?;
    /// #         }
    /// #         Ok(())
    /// #     }
    /// #
    /// #    fn get_fit_score(&self) -> f64 {
    /// #        self.models
    /// #            .iter()
    /// #            .max_by(|m1, m2| m1.fit_score.partial_cmp(&m2.fit_score).unwrap())
    /// #            .unwrap()
    /// #            .fit_score
    /// #    }
    /// #
    /// #    fn calculate_scores_and_trim(&mut self) -> Result<(), String> {
    /// #        self.models.sort_by(|a, b| a.fit_score.partial_cmp(&b.fit_score).unwrap().reverse());
    /// #        self.models.truncate(3);
    /// #        Ok(())
    /// #    }
    ///     //...
    ///
    ///     fn mutate(&mut self) -> Result<(), String> {
    ///         loop {
    ///             if self.models.len() < self.population_size.try_into().unwrap()
    ///             {
    ///                 self.models.push(mutate_random_individuals(&self.models))
    ///             }
    ///             else{
    ///                 return Ok(());
    ///             }
    ///         }
    ///     }
    /// }
    ///
    /// # fn main() -> Result<(), String> {
    /// let mut node = Node::initialize()?;
    /// assert_eq!(node.models.len(), 5);
    ///
    /// node.simulate(5)?;
    /// node.calculate_scores_and_trim()?;
    /// assert_eq!(node.models.len(), 3);
    ///
    /// node.mutate()?;
    /// assert_eq!(node.models.len(), 5);
    ///
    /// # assert_eq!(node.get_fit_score(), 9.0);
    /// # Ok(())
    /// # }
    /// ```
    fn mutate(&mut self) -> Result<(), String>;
}

/// Used externally to wrap a node implementing the [`GeneticNode`] trait. Processes state transitions for the given node as
/// well as signal recovery. Transition states are given by [`GeneticState`]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GeneticNodeWrapper<T>
where
    T: GeneticNode,
{
    pub data: Option<T>,
    state: GeneticState,
    pub iteration: u32,
}

impl<T> GeneticNodeWrapper<T>
where
    T: GeneticNode + fmt::Debug,
{
    /// Initializes a wrapper around a GeneticNode. If the initialization is successful the internal state will be changed to
    /// `GeneticState::Simulate` otherwise it will remain as `GeneticState::Initialize` and will attempt to be created in
    /// [`process_node`](#method.process_node).
    ///
    /// # Examples
    /// ```
    /// # use gemla::bracket::genetic_node::GeneticNode;
    /// # use gemla::bracket::genetic_node::GeneticNodeWrapper;
    /// # #[derive(Debug)]
    /// struct Node {
    /// #    pub fit_score: f64,
    ///     //...
    /// }
    ///
    /// impl GeneticNode for Node {
    ///     //...
    /// #    fn initialize() -> Result<Box<Self>, String> {
    /// #        Ok(Box::new(Node {fit_score: 0.0}))
    /// #    }
    /// #
    /// #
    /// #   fn simulate(&mut self, iterations: u64) -> Result<(), String> {
    /// #       Ok(())
    /// #   }
    /// #
    /// #   fn get_fit_score(&self) -> f64 {
    /// #       self.fit_score
    /// #   }
    /// #
    /// #   fn calculate_scores_and_trim(&mut self) -> Result<(), String> {
    /// #       Ok(())
    /// #   }
    /// #
    /// #   fn mutate(&mut self) -> Result<(), String> {
    /// #       Ok(())
    /// #   }
    /// }
    ///
    /// # fn main() -> Result<(), String> {
    /// let mut wrapped_node = GeneticNodeWrapper::<Node>::new()?;
    /// assert_eq!(wrapped_node.data.unwrap().get_fit_score(), 0.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> Result<Self, String> {
        let mut node = GeneticNodeWrapper {
            data: None,
            state: GeneticState::Initialize,
            iteration: 0,
        };

        let new_data = T::initialize()?;
        node.data = Some(*new_data);
        node.state = GeneticState::Simulate;

        Ok(node)
    }

    /// Performs state transitions on the [`GeneticNode`] wrapped by the [`GeneticNodeWrapper`].
    /// Will loop through the node training and scoring process for the given number of `iterations`.
    ///
    /// ## Transitions
    /// - `GeneticState::Initialize`: will attempt to call [`initialize`] on the node. When done successfully will change
    ///     the state to `GeneticState::Simulate`
    /// - `GeneticState::Simulate`: Will call [`simulate`] with a number of iterations (not for `iterations`). Will change the state to `GeneticState::Score`
    /// - `GeneticState::Score`: Will call [`calculate_scores_and_trim`] and when the number of `iterations` have been reached will change
    ///     state to `GeneticState::Finish`, otherwise it will change the state to `GeneticState::Mutate.
    /// - `GeneticState::Mutate`: Will call [`mutate`] and will change the state to `GeneticState::Simulate.`
    /// - `GeneticState::Finish`: Will finish processing the node and return.
    ///
    /// [`initialize`]: crate::bracket::genetic_node::GeneticNode#tymethod.initialize
    /// [`simulate`]: crate::bracket::genetic_node::GeneticNode#tymethod.simulate
    /// [`calculate_scores_and_trim`]: crate::bracket::genetic_node::GeneticNode#tymethod.calculate_scores_and_trim
    /// [`mutate`]: crate::bracket::genetic_node::GeneticNode#tymethod.mutate
    pub fn process_node(&mut self, iterations: u32) -> Result<(), String> {
        // Looping through each state transition until the number of iterations have been reached.
        loop {
            match (self.state, self.data.as_ref()) {
                (GeneticState::Initialize, _) => {
                    self.iteration = 0;
                    let new_data =
                        T::initialize().map_err(|e| format!("Error initializing node: {}", e))?;
                    self.data = Some(*new_data);
                    self.state = GeneticState::Simulate;
                }
                (GeneticState::Simulate, Some(_)) => {
                    self.data
                        .as_mut()
                        .unwrap()
                        .simulate(5)
                        .map_err(|e| format!("Error simulating node: {}", e))?;
                    self.state = GeneticState::Score;
                }
                (GeneticState::Score, Some(_)) => {
                    self.data
                        .as_mut()
                        .unwrap()
                        .calculate_scores_and_trim()
                        .map_err(|e| format!("Error scoring and trimming node: {}", e))?;

                    self.state = if self.iteration == iterations {
                        GeneticState::Finish
                    } else {
                        GeneticState::Mutate
                    }
                }
                (GeneticState::Mutate, Some(_)) => {
                    self.data
                        .as_mut()
                        .unwrap()
                        .mutate()
                        .map_err(|e| format!("Error mutating node: {}", e))?;
                    self.state = GeneticState::Simulate;
                }
                (GeneticState::Finish, Some(_)) => {
                    break;
                }
                _ => return Err(format!("Error processing node {:?}", self.data)),
            }
        }

        Ok(())
    }
}
