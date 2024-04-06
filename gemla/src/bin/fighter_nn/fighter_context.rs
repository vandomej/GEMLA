use std::sync::Arc;

use serde::ser::SerializeTuple;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tokio::sync::Semaphore;

const SHARED_SEMAPHORE_CONCURRENCY_LIMIT: usize = 50;
const VISIBLE_SIMULATIONS_CONCURRENCY_LIMIT: usize = 1;

#[derive(Debug, Clone)]
pub struct FighterContext {
    pub shared_semaphore: Arc<Semaphore>,
    pub visible_simulations: Arc<Semaphore>,
}

impl Default for FighterContext {
    fn default() -> Self {
        FighterContext {
            shared_semaphore: Arc::new(Semaphore::new(SHARED_SEMAPHORE_CONCURRENCY_LIMIT)),
            visible_simulations: Arc::new(Semaphore::new(VISIBLE_SIMULATIONS_CONCURRENCY_LIMIT)),
        }
    }
}

// Custom serialization to just output the concurrency limit.
impl Serialize for FighterContext {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Assuming the semaphore's available permits represent the concurrency limit.
        // This part is tricky since Semaphore does not expose its initial permits.
        // You might need to store the concurrency limit as a separate field if this assumption doesn't hold.
        let concurrency_limit = SHARED_SEMAPHORE_CONCURRENCY_LIMIT;
        let visible_concurrency_limit = VISIBLE_SIMULATIONS_CONCURRENCY_LIMIT;
        // serializer.serialize_u64(concurrency_limit as u64)

        // Serialize the concurrency limit as a tuple
        let mut state = serializer.serialize_tuple(2)?;
        state.serialize_element(&concurrency_limit)?;
        state.serialize_element(&visible_concurrency_limit)?;
        state.end()
    }
}

// Custom deserialization to reconstruct the FighterContext from a concurrency limit.
impl<'de> Deserialize<'de> for FighterContext {
    fn deserialize<D>(_: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize the tuple
        Ok(FighterContext {
            shared_semaphore: Arc::new(Semaphore::new(SHARED_SEMAPHORE_CONCURRENCY_LIMIT)),
            visible_simulations: Arc::new(Semaphore::new(VISIBLE_SIMULATIONS_CONCURRENCY_LIMIT)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization() {
        let context = FighterContext::default();
        let serialized = serde_json::to_string(&context).unwrap();
        let deserialized: FighterContext = serde_json::from_str(&serialized).unwrap();
        assert_eq!(
            context.shared_semaphore.available_permits(),
            deserialized.shared_semaphore.available_permits()
        );
        assert_eq!(
            context.visible_simulations.available_permits(),
            deserialized.visible_simulations.available_permits()
        );
    }
}
