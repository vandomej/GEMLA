pub mod genetic_node;
pub mod genetic_state;

use super::file_linked::FileLinked;
use super::tree;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use std::string::ToString;

#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
#[serde(tag = "enumType", content = "enumContent")]
pub enum IterationScaling {
    Linear(u32),
}

impl Default for IterationScaling {
    fn default() -> Self {
        IterationScaling::Linear(1)
    }
}

impl fmt::Display for IterationScaling {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            serde_json::to_string(self).expect("Unable to deserialize IterationScaling struct")
        )
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Bracket<T> {
    tree: tree::Tree<T>,
    step: u64,
    iteration_scaling: IterationScaling,
}

impl<T: fmt::Display + Serialize> fmt::Display for Bracket<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            serde_json::to_string(self).expect("Unable to deserialize Bracket struct")
        )
    }
}

impl<T> Bracket<T>
where
    T: genetic_node::GeneticNode
        + ToString
        + FromStr
        + Default
        + fmt::Display
        + DeserializeOwned
        + Serialize
        + Clone,
{
    pub fn initialize(file_path: String) -> Result<FileLinked<Self>, String> {
        FileLinked::new(
            Bracket {
                tree: btree!(T::initialize()),
                step: 0,
                iteration_scaling: IterationScaling::default(),
            },
            file_path,
        )
    }

    pub fn iteration_scaling(&mut self, iteration_scaling: IterationScaling) -> &mut Self {
        self.iteration_scaling = iteration_scaling;
        self
    }

    pub fn create_new_branch(&self, height: u64) -> tree::Tree<T> {
        if height == 1 {
            let mut base_node = btree!(T::initialize());

            base_node.val.simulate(match self.iteration_scaling {
                IterationScaling::Linear(x) => (x as u64) * height,
            });

            btree!(base_node.val)
        } else {
            let left = self.create_new_branch(height - 1);
            let right = self.create_new_branch(height - 1);
            let mut new_val = if left.val.get_fit_score() >= right.val.get_fit_score() {
                left.val.clone()
            } else {
                right.val.clone()
            };

            new_val.simulate(match self.iteration_scaling {
                IterationScaling::Linear(x) => (x as u64) * height,
            });

            btree!(new_val, left, right)
        }
    }

    pub fn run_simulation_step(&mut self) -> &mut Self {
        let new_branch = self.create_new_branch(self.step + 1);

        self.tree.val.simulate(match self.iteration_scaling {
            IterationScaling::Linear(x) => ((x as u64) * (self.step + 1)),
        });

        let new_val = if new_branch.val.get_fit_score() >= self.tree.val.get_fit_score() {
            new_branch.val.clone()
        } else {
            self.tree.val.clone()
        };

        self.tree = btree!(new_val, new_branch, self.tree.clone());

        self.step += 1;

        self
    }
}
