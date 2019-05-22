use uuid::Uuid;
use std::fmt;

pub struct State {
    id: Uuid
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.id)
    }
}

pub fn create() -> State {
    State {
        id: Uuid::new_v4()
    }
}