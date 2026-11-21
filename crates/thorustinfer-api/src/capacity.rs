//! A managed HTTP or batch process publishes the same worker-owned capacity snapshot.
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thorustinfer_engine::{execution::ContextPlan, model::Model};

pub(crate) const DIRECTORY: &str = "THORUSTINFER_WORKER_PLAN_DIR";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub(crate) struct Snapshot {
    pub effective: usize,
    pub planned: usize,
    pub remaining: usize,
}
pub(crate) fn path(directory: &Path) -> PathBuf {
    directory.join("capacity.json")
}
pub(crate) fn read(path: &Path) -> Result<Option<Snapshot>, String> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|e| e.to_string()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}
pub(crate) struct Publisher {
    plan: ContextPlan,
    file: Option<PathBuf>,
    previous: Option<Snapshot>,
}
impl Publisher {
    pub(crate) fn new(plan: ContextPlan) -> Self {
        Self {
            plan,
            file: std::env::var_os(DIRECTORY).map(|d| path(Path::new(&d))),
            previous: None,
        }
    }
    pub(crate) fn update(&mut self, model: &Model, features: usize) -> Result<(), String> {
        model.refresh_context_growth(&self.plan, features);
        let snapshot = Snapshot {
            effective: self.plan.effective,
            planned: self.plan.planned_growth_bytes,
            remaining: self.plan.remaining_growth_bytes(),
        };
        if self.previous != Some(snapshot)
            && let Some(file) = &self.file
        {
            let temp = file.with_extension("tmp");
            std::fs::write(
                &temp,
                serde_json::to_vec(&snapshot).map_err(|e| e.to_string())?,
            )
            .and_then(|()| std::fs::rename(temp, file))
            .map_err(|e| e.to_string())?;
        }
        self.previous = Some(snapshot);
        Ok(())
    }
}
