//! Device and execution contracts. This crate never loads an accelerator driver.
use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

pub type Result<T> = std::result::Result<T, String>;

/// Process-local ownership of device allocations. An ordinal alone does not
/// distinguish a restarted worker or two sessions on the same device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceOwner {
    backend: Backend,
    ordinal: u32,
    generation: u64,
}
impl DeviceOwner {
    pub fn new(backend: Backend, ordinal: u32) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            backend,
            ordinal,
            generation: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }
    pub fn require(self, owner: Self) -> Result<()> {
        if self != owner {
            return Err(
                "Device allocation belongs to a different backend, device or session".into(),
            );
        }
        Ok(())
    }
}

/// CUDA serialization retains its historical flat report fields.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum DeviceInfo {
    Cuda {
        name: String,
        sm: [i32; 2],
        driver_version: i32,
        total_bytes: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Cuda,
}
impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Cuda => "cuda",
        })
    }
}
impl FromStr for Backend {
    type Err = String;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "cuda" => Ok(Self::Cuda),
            _ => Err(format!("Unknown device backend {value:?}; expected cuda")),
        }
    }
}

/// Host ABI and device architecture are independent.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub host_arch: String,
    pub backend: Backend,
    pub device_arch: String,
}
/// CUDA device capability to native package target (architecture-specific cubins).
pub fn cuda_arch(sm: [i32; 2]) -> Option<&'static str> {
    match sm {
        [8, 7] => Some("sm_87"),
        [11, 0] => Some("sm_110a"),
        _ => None,
    }
}

impl Target {
    pub fn validate(&self) -> Result<()> {
        if !matches!(self.host_arch.as_str(), "aarch64" | "x86_64") {
            return Err("Unsupported execution-package host architecture".into());
        }
        match (self.backend, self.device_arch.as_str()) {
            (Backend::Cuda, "sm_87" | "sm_110a") => Ok(()),
            _ => Err("Device architecture differs from its backend or is unsupported".into()),
        }
    }
    /// Existing schema-1 packages retain their original target strings.
    pub fn from_package(value: &str) -> Result<Self> {
        let (backend, device_arch, host_arch) = match value {
            "sm_87" | "sm_110a" => (Backend::Cuda, value, "aarch64"),
            _ => return Err(format!("Unsupported execution-package target {value:?}")),
        };
        Ok(Self {
            host_arch: host_arch.into(),
            backend,
            device_arch: device_arch.into(),
        })
    }
    pub fn require_cuda_device(&self, sm: [i32; 2]) -> Result<()> {
        if self.backend == Backend::Cuda && cuda_arch(sm) == Some(self.device_arch.as_str()) {
            return Ok(());
        }
        Err(format!(
            "Package requires {}, CUDA device has SM {}.{}",
            self.device_arch, sm[0], sm[1]
        ))
    }
    pub fn require_host(&self, host: &str) -> Result<()> {
        self.validate()?;
        if self.host_arch != host {
            return Err(format!(
                "Execution package requires {} host; running on {host}",
                self.host_arch
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphMode {
    #[default]
    DecodeOnly,
    Full,
    Off,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionPhase {
    Prefill,
    Decode,
    Vision,
}
impl GraphMode {
    pub fn uses_graph(self, phase: ExecutionPhase) -> bool {
        match self {
            Self::DecodeOnly => phase == ExecutionPhase::Decode,
            Self::Full => true,
            Self::Off => false,
        }
    }
}
impl fmt::Display for GraphMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::DecodeOnly => "decode_only",
            Self::Full => "full",
            Self::Off => "off",
        })
    }
}
impl FromStr for GraphMode {
    type Err = String;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "decode_only" => Ok(Self::DecodeOnly),
            "full" => Ok(Self::Full),
            "off" => Ok(Self::Off),
            _ => Err(format!(
                "Invalid graph mode {value:?}; expected decode_only, full or off"
            )),
        }
    }
}

/// Logical extents do not assume a CUDA or CANN physical mapping granularity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryRange {
    pub offset: usize,
    pub bytes: usize,
}
impl MemoryRange {
    pub fn end(&self) -> Result<usize> {
        self.offset
            .checked_add(self.bytes)
            .ok_or_else(|| "Memory extent overflow".into())
    }
    pub fn within(&self, capacity: usize) -> Result<()> {
        if self.end()? > capacity {
            return Err(format!(
                "Memory extent exceeds allocation: offset={}, bytes={}, capacity={capacity}",
                self.offset, self.bytes
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cuda_targets_require_matching_device() {
        let thor = Target::from_package("sm_110a").unwrap();
        assert!(thor.require_cuda_device([11, 0]).is_ok());
        assert!(thor.require_cuda_device([8, 7]).is_err());
        assert!(
            Target::from_package("sm_87")
                .unwrap()
                .require_cuda_device([11, 0])
                .is_err()
        );
    }
    #[test]
    fn cuda_report_preserves_legacy_flat_fields() {
        let report = DeviceInfo::Cuda {
            name: "Orin".into(),
            sm: [8, 7],
            driver_version: 12060,
            total_bytes: 64 << 30,
        };
        assert_eq!(
            serde_json::to_value(report).unwrap(),
            serde_json::json!({
                "name":"Orin", "sm":[8,7], "driver_version":12060, "total_bytes":64u64 << 30,
            })
        );
    }
    #[test]
    fn target_rejects_mixed_backends_and_host_abis() {
        assert_eq!(
            Target::from_package("sm_87").unwrap().backend,
            Backend::Cuda
        );
        assert!(Target::from_package("ascend950").is_err());
    }
    #[test]
    fn extents_detect_overflow_and_out_of_bounds() {
        assert!(
            MemoryRange {
                offset: usize::MAX,
                bytes: 1
            }
            .end()
            .is_err()
        );
        assert!(
            MemoryRange {
                offset: 12,
                bytes: 5
            }
            .within(16)
            .is_err()
        );
        assert!(
            MemoryRange {
                offset: 12,
                bytes: 4
            }
            .within(16)
            .is_ok()
        );
    }
    #[test]
    fn snapshots_cannot_cross_devices_or_worker_incarnations() {
        let first = DeviceOwner::new(Backend::Cuda, 0);
        assert!(first.require(first).is_ok());
        assert!(first.require(DeviceOwner::new(Backend::Cuda, 0)).is_err());
        assert!(first.require(DeviceOwner::new(Backend::Cuda, 1)).is_err());
    }
}
