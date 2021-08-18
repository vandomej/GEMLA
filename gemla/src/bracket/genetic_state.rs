//! An enum used to control the state of a [`GeneticNode`]
//! 
//! [`GeneticNode`]: crate::bracket::genetic_node

use serde::{Deserialize, Serialize};

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
