//! Geometry of the two version-1 host operations in the Thor 4B execution package.
use crate::artifact::{Access, Argument, Dtype, Kernel, Nvfp4Operation, Result};
use crate::model::Manifest;

pub(crate) fn validate(m: &Manifest, k: &Kernel, op: Nvfp4Operation) -> Result<()> {
    let err = || format!("{}: invalid SM110a NVFP4 FFN v1 contract", k.name);
    let rest = k.name.strip_prefix("qwen_rows/layer").ok_or_else(err)?;
    let (index, tail) = rest.split_once("/post/").ok_or_else(err)?;
    let layer = index.parse::<usize>().map_err(|_| err())?;
    let gate = op == Nvfp4Operation::GateUp;
    let operation = if gate { "gate_up" } else { "down" };
    if m.target != "sm_110a"
        || layer >= 32
        || tail.split('/').next() != Some(operation)
        || k.symbol != "orinfer_nvfp4_ffn_v1"
        || k.readonly_arguments.as_slice()
            != if gate {
                &[
                    true, true, true, true, false, false, true, true, true, true, true, false,
                ][..]
            } else {
                &[true, true, true, true, false, false, true, true][..]
            }
        || k.args.len() != if gate { 12 } else { 8 }
    {
        return Err(err());
    }
    let check = |i: usize,
                 name: &str,
                 dtype: Dtype,
                 shape: &[usize],
                 access: Access,
                 offset: Option<usize>|
     -> Result<()> {
        let matches = match (&k.args[i], offset) {
            (Argument::Buffer { name: n }, None) => n == name,
            (Argument::BufferSlice { name: n, offset: a }, Some(b)) => n == name && *a == b,
            _ => false,
        };
        if !matches
            || !m.buffers.iter().any(|b| {
                b.name == name
                    && b.dtype == dtype
                    && b.shape == shape
                    && b.access == access
                    && b.alignment >= 16
            })
        {
            return Err(err());
        }
        Ok(())
    };
    let rw = Access::ReadWrite;
    let weight = |start: usize, role: &str| -> Result<()> {
        for (j, part, dtype, len) in [
            (0, "W", Dtype::U8, 2560 * 9216 / 2),
            (1, "S", Dtype::U8, 2560 * 9216 / 16),
            (2, "G", Dtype::F32, 1),
        ] {
            check(
                start + j,
                &format!("NVFP4.{layer}.{role}.{part}"),
                dtype,
                &[len],
                Access::Read,
                None,
            )?;
        }
        Ok(())
    };
    if !matches!(k.args[6], Argument::I32 { value: 1..=4096 }) {
        return Err(err());
    }
    check(5, "NVFP4.Scratch", Dtype::U8, &[64 << 20], rw, None)?;
    if gate {
        check(0, "Norm", Dtype::F16, &[4096, 2560], rw, None)?;
        weight(1, "gate")?;
        check(4, "GateUp", Dtype::F16, &[8192, 9216], rw, None)?;
        check(
            7,
            "orinfer.rotation.k2560.signs",
            Dtype::I8,
            &[2560],
            Access::Read,
            None,
        )?;
        weight(8, "up")?;
        check(
            11,
            "GateUp",
            Dtype::F16,
            &[8192, 9216],
            rw,
            Some(4096 * 9216 * 2),
        )?;
    } else {
        check(0, "RotationValues", Dtype::F16, &[4096, 9216], rw, None)?;
        weight(1, "down")?;
        check(4, "Hidden", Dtype::F16, &[4096, 2560], rw, None)?;
        check(7, "RotationMaxima", Dtype::F32, &[4096, 36], rw, None)?;
    }
    Ok(())
}
