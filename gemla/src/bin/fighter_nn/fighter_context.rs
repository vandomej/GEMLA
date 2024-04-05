use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tokio::sync::Semaphore;

const SHARED_SEMAPHORE_CONCURRENCY_LIMIT: usize = 20;


#[derive(Debug, Clone)]
pub struct FighterContext {
    pub shared_semaphore: Arc<Semaphore>,
}

impl Default for FighterContext {
    fn default() -> Self {
        FighterContext {
            shared_semaphore: Arc::new(Semaphore::new(SHARED_SEMAPHORE_CONCURRENCY_LIMIT)),
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
        serializer.serialize_u64(concurrency_limit as u64)
    }
}

// Custom deserialization to reconstruct the FighterContext from a concurrency limit.
impl<'de> Deserialize<'de> for FighterContext {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let concurrency_limit = u64::deserialize(deserializer)?;
        Ok(FighterContext {
            shared_semaphore: Arc::new(Semaphore::new(concurrency_limit as usize)),
        })
    }
}