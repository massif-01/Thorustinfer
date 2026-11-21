//! Resident BF16 draft matrices and the SM110a projection launch contract.
use crate::artifact::Result;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::bf16;
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path, sync::Arc};

#[derive(Deserialize)]
struct Package {
    schema: u32,
    verifier: (String, String),
    target: String,
    config_sha256: String,
    weight_sha256: String,
    projections: Vec<Projection>,
}
#[derive(Deserialize)]
struct Projection {
    n: usize,
    k: usize,
    cubin: String,
    abi: Abi,
}
#[derive(Deserialize)]
struct Abi {
    symbol: String,
    launch_expressions: BTreeMap<String, String>,
    ordered_arguments: Vec<BTreeMap<String, String>>,
}
struct Matrix {
    n: usize,
    k: usize,
    data: CudaSlice<u16>,
}
pub(super) struct Store {
    stream: Arc<CudaStream>,
    kernels: BTreeMap<(usize, usize), (CudaFunction, LaunchConfig)>,
    matrices: BTreeMap<String, Matrix>,
    host: BTreeMap<String, (Vec<usize>, Vec<u16>)>,
}
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
impl Store {
    pub(super) fn load(
        root: &Path,
        config: &[u8],
        checkpoint: &(String, String),
        verify: bool,
    ) -> Result<Self> {
        let package: Package = crate::model::read(&root.join("draft.json"))?;
        if &package.verifier != checkpoint {
            return Err("Block draft package is pinned to a different verifier checkpoint".into());
        }
        if package.schema != 1
            || package.target != "sm_110a"
            || crate::artifact::sha256(config) != package.config_sha256
        {
            return Err("Incompatible draft package or configuration digest".into());
        }
        let ctx = CudaContext::new(0).map_err(err)?;
        let stream = ctx.new_stream().map_err(err)?;
        let mut kernels = BTreeMap::new();
        for p in package.projections {
            if p.cubin.len() != 64 || !p.cubin.bytes().all(|x| x.is_ascii_hexdigit()) {
                return Err("Invalid projection identity".into());
            }
            let bytes = std::fs::read(root.join(format!("{}.cubin", p.cubin))).map_err(err)?;
            let arg_names = p
                .abi
                .ordered_arguments
                .iter()
                .map(|a| {
                    (
                        a.get("value").map(String::as_str),
                        a.get("ctype").map(String::as_str),
                    )
                })
                .collect::<Vec<_>>();
            if crate::artifact::sha256(&bytes) != p.cubin
                || p.n % 64 != 0
                || p.k % 32 != 0
                || p.n == 0
                || p.k == 0
                || arg_names
                    != ["A.data_ptr()", "W.data_ptr()", "Y.data_ptr()"]
                        .map(|v| (Some(v), Some("ctypes.c_void_p")))
            {
                return Err("Invalid BF16 draft projection ABI".into());
            }
            let dim = |k: &str| -> Result<u32> {
                p.abi
                    .launch_expressions
                    .get(k)
                    .ok_or("Missing projection launch dimension")?
                    .parse::<u32>()
                    .map_err(err)
            };
            let launch = LaunchConfig {
                grid_dim: (dim("gridDimX")?, dim("gridDimY")?, dim("gridDimZ")?),
                block_dim: (dim("blockDimX")?, dim("blockDimY")?, dim("blockDimZ")?),
                shared_mem_bytes: dim("sharedMemBytes")?,
            };
            if launch.grid_dim != (u32::try_from(p.n / 64).map_err(err)?, 1, 1)
                || launch.block_dim != (128, 1, 1)
            {
                return Err("Invalid draft projection launch geometry".into());
            }
            let module = ctx
                .load_module(cudarc::nvrtc::Ptx::from_binary(bytes))
                .map_err(err)?;
            let f = module.load_function(&p.abi.symbol).map_err(err)?;
            kernels.insert((p.n, p.k), (f, launch));
        }
        let file = std::fs::File::open(root.join("model.safetensors")).map_err(err)?;
        // SAFETY: The installed checkpoint stays immutable throughout initialization.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(err)?;
        if verify && crate::artifact::sha256(&mmap) != package.weight_sha256 {
            return Err("Draft weight digest mismatch".into());
        }
        let tensors = safetensors::SafeTensors::deserialize(&mmap).map_err(err)?;
        let mut matrices = BTreeMap::new();
        let mut host = BTreeMap::new();
        for (name, v) in tensors.tensors() {
            if v.dtype() != safetensors::Dtype::BF16 {
                return Err(format!(
                    "Expected BF16 draft tensor {name}, got {:?}",
                    v.dtype()
                ));
            }
            let values = v
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| u16::from_le_bytes(b))
                .collect::<Vec<_>>();
            if let [n, k] = v.shape()
                && name.ends_with(".weight")
                && !name.starts_with("confidence_head.")
                && !name.starts_with("markov_head.markov_w1")
            {
                if !kernels.contains_key(&(*n, *k)) {
                    return Err(format!("Missing draft projection for {name}"));
                }
                let data = stream.clone_htod(&values).map_err(err)?;
                matrices.insert(name, Matrix { n: *n, k: *k, data });
            } else {
                host.insert(name, (v.shape().to_vec(), values));
            }
        }
        stream.synchronize().map_err(err)?;
        Ok(Self {
            stream,
            kernels,
            matrices,
            host,
        })
    }
    pub(super) fn host(&self, name: &str) -> Result<&[u16]> {
        Ok(&self
            .host
            .get(name)
            .ok_or_else(|| format!("Missing draft tensor {name}"))?
            .1)
    }
    pub(super) fn check(&self, name: &str, shape: &[usize]) -> Result<()> {
        let valid = self.host.get(name).is_some_and(|t| t.0 == shape)
            || self.matrices.get(name).is_some_and(|m| shape == [m.n, m.k]);
        if valid {
            Ok(())
        } else {
            Err(format!("Draft tensor {name} must have shape {shape:?}"))
        }
    }
    pub(super) fn linear(&self, name: &str, input: &[f32]) -> Result<Vec<f32>> {
        let m = self
            .matrices
            .get(name)
            .ok_or_else(|| format!("Missing draft matrix {name}"))?;
        if !input.len().is_multiple_of(m.k) {
            return Err(format!("Bad input width for {name}"));
        }
        let (f, launch) = &self.kernels[&(m.n, m.k)];
        let mut values = Vec::with_capacity(input.len() / m.k * m.n);
        for block in input.chunks(16 * m.k) {
            let mut x = block
                .iter()
                .map(|&v| bf16::from_f32(v).to_bits())
                .collect::<Vec<_>>();
            x.resize(16 * m.k, 0);
            let x = self.stream.clone_htod(&x).map_err(err)?;
            let mut y = self.stream.alloc_zeros::<u16>(16 * m.n).map_err(err)?;
            let mut call = self.stream.launch_builder(f);
            call.arg(&x).arg(&m.data).arg(&mut y);
            // SAFETY: Package ABI, grid, matrix shapes and the padded 16-row extents were checked.
            unsafe { call.launch(*launch) }.map_err(err)?;
            let result = self.stream.clone_dtoh(&y).map_err(err)?;
            values.extend(
                result[..block.len() / m.k * m.n]
                    .iter()
                    .map(|&v| bf16::from_bits(v).to_f32()),
            );
        }
        Ok(values)
    }
}
