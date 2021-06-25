use super::super::bracket;

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use std::string::ToString;

#[derive(Default, Deserialize, Serialize, Clone)]
struct TestState {
    pub score: f64,
}

impl FromStr for TestState {
    type Err = String;

    fn from_str(s: &str) -> Result<TestState, Self::Err> {
        toml::from_str(s).map_err(|_| format!("Unable to parse string {}", s))
    }
}

impl fmt::Display for TestState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.score)
    }
}

impl TestState {
    fn new(score: f64) -> TestState {
        TestState { score: score }
    }
}

impl bracket::genetic_state::GeneticState for TestState {
    fn run_simulation(&mut self, iterations: u64) {
        self.score += iterations as f64;
    }

    fn get_fit_score(&self) -> f64 {
        self.score
    }

    fn initialize() -> Self {
        TestState { score: 0.0 }
    }
}

#[test]
fn test_new() {
    let bracket = bracket::Bracket::<TestState>::initialize("./temp".to_string())
        .expect("Bracket failed to initialize");

    assert_eq!(
        format!("{}", bracket),
        format!("{{\"tree\":{},\"step\":0,\"iteration_scaling\":{{\"enumType\":\"Linear\",\"enumContent\":1}}}}", 
        btree!(TestState::new(0.0)))
    );

    std::fs::remove_file("./temp").expect("Unable to remove file");
}

#[test]
fn test_run() {
    let mut bracket = bracket::Bracket::<TestState>::initialize("./temp".to_string())
        .expect("Bracket failed to initialize");

    bracket
        .mutate(|b| drop(b.iteration_scaling(bracket::IterationScaling::Linear(2))))
        .expect("Failed to set iteration scaling");

    for _ in 0..3 {
        bracket
            .mutate(|b| drop(b.run_simulation_step()))
            .expect("Failed to run step");
    }

    assert_eq!(
        format!("{}", bracket),
        format!("{{\"tree\":{},\"step\":3,\"iteration_scaling\":{{\"enumType\":\"Linear\",\"enumContent\":2}}}}", 
        btree!(
            TestState::new(12.0),
            btree!(
                TestState::new(12.0),
                btree!(TestState::new(6.0),
                    btree!(TestState::new(2.0)),
                    btree!(TestState::new(2.0))),
                btree!(TestState::new(6.0),
                    btree!(TestState::new(2.0)),
                    btree!(TestState::new(2.0)))
            ),
            btree!(
                TestState::new(12.0),
                btree!(TestState::new(6.0),
                    btree!(TestState::new(2.0)),
                    btree!(TestState::new(2.0))),
                btree!(TestState::new(6.0),
                    btree!(TestState::new(2.0)),
                    btree!(TestState::new(2.0))))
            )
        )
    );

    std::fs::remove_file("./temp").expect("Unable to remove file");
}
