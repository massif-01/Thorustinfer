//! Trusted, engine-relative SM110a CUTLASS provider. A model declares an operation, never a host library path.
use super::{Driver, Kernel, Result, Value, sys};
use libloading::Library;
use thorustinfer_model_sdk::artifact::Nvfp4Operation;

type NativeLaunch = unsafe extern "C" fn(i32, i32, *const u64, sys::CUstream) -> i32;

pub(super) struct Provider {
    _library: Library,
    pub(super) launch: NativeLaunch,
}
impl Provider {
    pub(super) fn open() -> Result<Self> {
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let root = exe
            .parent()
            .and_then(|p| p.parent())
            .ok_or("Invalid engine installation")?;
        let path = root.join("lib/libthorustinfer_nvfp4_v1.so");
        // SAFETY: The installed engine owns this fixed library and versioned C ABI.
        // Catalog retains it for all captured work; Session is dropped before Catalog.
        let library = unsafe { Library::new(&path) }
            .map_err(|e| format!("Load NVFP4 provider {}: {e}", path.display()))?;
        // SAFETY: The v1 symbol in the engine-owned library has the fixed NativeLaunch signature.
        let launch = *unsafe { library.get::<NativeLaunch>(b"orinfer_nvfp4_ffn_v1\0") }
            .map_err(|e| e.to_string())?;
        Ok(Self {
            _library: library,
            launch,
        })
    }
}

#[derive(Clone, Copy)]
pub(super) enum ModelFunction {
    Cuda(sys::CUfunction),
    Native(Nvfp4Operation, NativeLaunch),
}

pub(super) fn launch(
    k: &Kernel,
    f: ModelFunction,
    args: &mut [Value],
    driver: &Driver,
    stream: sys::CUstream,
) -> Result<()> {
    let (operation, native) = match f {
        ModelFunction::Native(operation, native) => (operation, native),
        ModelFunction::Cuda(handle) => {
            return super::launch_kernel(k, handle, args, driver, stream);
        }
    };
    let Some(Value::I32(rows)) = args.get(6).copied() else {
        return Err("Missing NVFP4 row count".into());
    };
    if !(1..=4096).contains(&rows) {
        return Err("NVFP4 rows exceed package capacity".into());
    }
    let mut pointers = [0u64; 11];
    let mut addresses = args.iter().enumerate().filter(|(i, _)| *i != 6);
    let gate = operation == Nvfp4Operation::GateUp;
    for slot in &mut pointers[..if gate { 11 } else { 7 }] {
        let Some((_, Value::Pointer(ptr))) = addresses.next() else {
            return Err("Missing NVFP4 device operand".into());
        };
        *slot = *ptr;
    }
    if addresses.next().is_some() {
        return Err("Extra NVFP4 device operand".into());
    }
    // SAFETY: The model validates fixed operand roles, ranges, access and dimensions.
    // The CUDA session owns all operands and orders both projections on this stream.
    let status = unsafe { native(i32::from(gate), rows, pointers.as_ptr(), stream) };
    if status == 0 {
        Ok(())
    } else {
        Err(format!("{}: NVFP4 launch failed ({status})", k.name))
    }
}
