use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
#[serde(tag = "enumType", content = "enumContent")]
pub enum GeneticState {
    Initialize,
    Simulate,
    Score,
    Mutate,
    Finish,
}
