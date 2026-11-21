//! Device implementation boundary for model execution and diagnostics.
pub(crate) mod arena;
pub(crate) mod cuda;
pub(crate) mod growth;
pub(crate) mod kv;
pub(crate) mod report;
pub(crate) mod snapshot;
mod values;
pub(crate) use cuda::executor::Executor;
pub(crate) use report::LoadStats;
pub use thorustinfer_device::{Backend, DeviceInfo, GraphMode, Target};
pub(crate) use values::floats;

/// Symbol presence is diagnostic information, not a hardware capability promise.
#[derive(Debug, serde::Serialize)]
pub struct Inventory {
    pub backend: Backend,
    pub host_arch: String,
    pub device_count: u32,
    pub soc: Option<String>,
    pub runtime_library: String,
    pub api_symbols: Vec<String>,
    pub model_execution_available: bool,
}
pub fn inspect(backend: Backend) -> crate::artifact::Result<Inventory> {
    match backend {
        Backend::Cuda => cuda::inspect(),
    }
}
