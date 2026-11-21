use serde::{Deserialize, Serialize};
pub type Result<T> = std::result::Result<T, String>;

/// Validate the canonical lowercase hexadecimal spelling of a SHA-256 digest.
pub fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FileIdentity {
    pub file: String,
    pub sha256: String,
    /// Named U8 asset in a safetensors bundle; absent for standalone files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor: Option<String>,
}

pub type AssetKey = (String, Option<String>);

impl FileIdentity {
    pub fn key(&self) -> AssetKey {
        (self.file.clone(), self.tensor.clone())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Dtype {
    U8,
    I8,
    U16,
    F16,
    Bf16,
    F32,
    F64,
    U32,
    I32,
    U64,
    I64,
}
impl Dtype {
    pub fn bytes(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::F16 | Self::Bf16 | Self::U16 => 2,
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Buffer<Data = FileIdentity> {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub layout: String,
    pub alignment: u64,
    pub access: Access,
    pub data: Option<Data>,
}
impl<Data> Buffer<Data> {
    pub fn bytes(&self) -> Result<usize> {
        if self.shape.is_empty() || self.shape.contains(&0) {
            return Err(format!("{}: empty shape", self.name));
        }
        self.shape.iter().try_fold(self.dtype.bytes(), |n, d| {
            n.checked_mul(*d)
                .ok_or_else(|| format!("{}: size overflow", self.name))
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Kernel {
    /// Absent in existing schema-1 packages: the original CUDA launch ABI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub launch: Option<Launch>,
    pub name: String,
    pub module: FileIdentity,
    pub source: FileIdentity,
    pub host_abi: FileIdentity,
    pub symbol: String,
    #[serde(default)]
    pub grid: [u32; 3],
    #[serde(default)]
    pub block: [u32; 3],
    #[serde(default)]
    pub shared_memory_bytes: u32,
    #[serde(default)]
    pub cooperative: bool,
    pub args: Vec<Argument>,
    /// Generated from the CUDA prototype. Unannotated kernels remain serial.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub readonly_arguments: Vec<bool>,
}

/// Native host wrappers interpret tiling buffers through the declared args ABI.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "backend", rename_all = "snake_case", deny_unknown_fields)]
pub enum Launch {
    ThorNvfp4 { operation: Nvfp4Operation },
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Nvfp4Operation {
    GateUp,
    Down,
}

impl Launch {
    pub fn backend(&self) -> thorustinfer_device::Backend {
        match self {
            Self::ThorNvfp4 { .. } => thorustinfer_device::Backend::Cuda,
        }
    }
}

pub(crate) fn validate_geometry(grid: Option<[u32; 3]>, block: [u32; 3]) -> Result<()> {
    if grid.is_some_and(|g| g.contains(&0))
        || block.contains(&0)
        || block
            .iter()
            .try_fold(1u32, |n, d| n.checked_mul(*d))
            .is_none_or(|n| n > 1024)
    {
        return Err("Invalid CUDA launch dimensions".into());
    }
    Ok(())
}
impl Kernel {
    pub fn backend(&self) -> thorustinfer_device::Backend {
        self.launch
            .as_ref()
            .map_or(thorustinfer_device::Backend::Cuda, Launch::backend)
    }
    pub fn validate_launch(&self, target: &thorustinfer_device::Target) -> Result<()> {
        target.validate()?;
        if self.backend() != target.backend {
            return Err("Kernel backend differs from package target".into());
        }
        validate_geometry(Some(self.grid), self.block)
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Argument {
    Buffer { name: String },
    BufferSlice { name: String, offset: usize },
    I32 { value: i32 },
    U32 { value: u32 },
    I64 { value: i64 },
    U64 { value: u64 },
    F32 { value: f32 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use thorustinfer_device::Target;

    fn kernel() -> serde_json::Value {
        let asset = serde_json::json!({"file":"kernel.bin","sha256":"0".repeat(64)});
        serde_json::json!({
            "name":"projection", "module":asset, "source":asset, "host_abi":asset,
            "symbol":"projection", "grid":[1,1,1], "block":[128,1,1],
            "shared_memory_bytes":0, "cooperative":false, "args":[]
        })
    }

    #[test]
    fn legacy_cuda_kernel_retains_its_serialized_contract() {
        let original = kernel();
        let parsed: Kernel = serde_json::from_value(original.clone()).unwrap();
        parsed
            .validate_launch(&Target::from_package("sm_87").unwrap())
            .unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), original);
    }

    #[test]
    fn cuda_launch_rejects_zero_and_overflowing_geometry() {
        for block in [[0, 1, 1], [1025, 1, 1], [u32::MAX, u32::MAX, 1]] {
            assert!(validate_geometry(Some([1; 3]), block).is_err());
        }
        assert!(validate_geometry(Some([0, 1, 1]), [32, 1, 1]).is_err());
    }
}
