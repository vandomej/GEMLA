pub mod genetic_state;

use super::file_linked::FileLinked;
use super::tree;

use std::fmt;
use std::str::FromStr;
use std::string::ToString;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "enumType", content = "enumContent")]
pub enum IterationScaling {
    Linear(u32)
}

impl Default for IterationScaling {
    fn default() -> Self {
        IterationScaling::Linear(1)
    }
}

impl fmt::Display for IterationScaling {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", serde_json::to_string(self).expect("Unable to deserialize IterationScaling struct"))
    }
}


#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Bracket<T> {
    tree: tree::Tree<T>,
    step: u32,
    iteration_scaling: IterationScaling
}

impl<T: fmt::Display + Serialize> fmt::Display for Bracket<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", serde_json::to_string(self).expect("Unable to deserialize Bracket struct"))
    }
}

impl<T> Bracket<T> 
    where T: genetic_state::GeneticState + ToString + FromStr + Default + fmt::Display + DeserializeOwned + Serialize + Clone
{
    pub fn initialize(file_path: String) -> Result<FileLinked<Self>, String>
    {
        FileLinked::new(
            Bracket
            {
                tree: btree!(T::initialize()),
                step: 0,
                iteration_scaling: IterationScaling::default()
            }
        ,file_path)
    }

    pub fn iteration_scaling(&mut self, iteration_scaling: IterationScaling) -> &mut Self
    {
        self.iteration_scaling = iteration_scaling;
        self
    }

    pub fn run_simulation_step(&mut self) -> &mut Self 
    {
        self.tree.val.run_simulation(
            match self.iteration_scaling
            {
                IterationScaling::Linear(x) => x
            }
        );

        let mut new_branch = btree!(T::initialize());
        new_branch.val.run_simulation(
            match self.iteration_scaling
            {
                IterationScaling::Linear(x) => x * (self.step + 1)
            }
        );

        self.tree = btree!(
            self.tree.val.clone(), 
            Some(self.tree.clone()),
            Some(new_branch)
        );
        self.step += 1;

        self
    }
}
