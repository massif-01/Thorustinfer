//! Shared bounded file reads for CUDA weight loading.
use super::TensorSource;
use crate::artifact::Result;
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::ErrorKind,
    os::unix::fs::{FileExt, OpenOptionsExt},
};
pub(crate) const ALIGNMENT: usize = 4096;
struct ReadFile {
    buffered: File,
    direct: Option<File>,
    size: usize,
}
#[derive(Default)]
pub(crate) struct Reader {
    files: BTreeMap<std::path::PathBuf, ReadFile>,
}
fn unsupported_direct(error: &std::io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(libc::EINVAL | libc::EOPNOTSUPP | libc::ENOSYS)
    )
}

impl Reader {
    /// Reuse shard descriptors and accept EOF tails on sector-aligned direct reads.
    pub(crate) fn read(
        &mut self,
        source: &TensorSource,
        offset: usize,
        bytes: usize,
        target: &mut [u8],
    ) -> Result<usize> {
        let absolute = source
            .offset
            .checked_add(offset)
            .ok_or("Weight offset overflow")?;
        if bytes == 0
            || bytes > target.len().saturating_sub(ALIGNMENT)
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > source.bytes)
        {
            return Err("Weight chunk exceeds tensor".into());
        }
        let aligned = absolute / ALIGNMENT * ALIGNMENT;
        let skip = absolute - aligned;
        let extent = (skip + bytes).next_multiple_of(ALIGNMENT);
        if extent > target.len() {
            return Err("Weight chunk exceeds staging slot".into());
        }
        if !self.files.contains_key(&source.path) && self.files.len() >= 16 {
            self.files.pop_first();
        }
        let file = match self.files.entry(source.path.clone()) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                let buffered = File::open(&source.path)
                    .map_err(|e| format!("{}: {e}", source.path.display()))?;
                let size = usize::try_from(buffered.metadata().map_err(|e| e.to_string())?.len())
                    .map_err(|_| "Weight shard length overflow")?;
                let direct = match OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(&source.path)
                {
                    Ok(file) => Some(file),
                    Err(error) if unsupported_direct(&error) => None,
                    Err(error) => return Err(format!("{}: {error}", source.path.display())),
                };
                entry.insert(ReadFile {
                    buffered,
                    direct,
                    size,
                })
            }
        };
        if absolute
            .checked_add(bytes)
            .is_none_or(|end| end > file.size)
        {
            return Err(format!(
                "{}: truncated weight payload",
                source.path.display()
            ));
        }
        if (target.as_ptr() as usize).is_multiple_of(ALIGNMENT)
            && let Some(direct) = &file.direct
        {
            let read = loop {
                match direct.read_at(&mut target[..extent], aligned as u64) {
                    Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                    result => break result,
                }
            };
            match read {
                Ok(n) if n == extent.min(file.size - aligned) => return Ok(skip),
                Ok(_) => {
                    return Err(format!(
                        "{}: short direct weight read",
                        source.path.display()
                    ));
                }
                Err(error) if unsupported_direct(&error) => {
                    file.direct = None;
                }
                Err(error) => return Err(format!("{}: {error}", source.path.display())),
            }
        }
        file.buffered
            .read_exact_at(&mut target[..bytes], absolute as u64)
            .map_err(|e| format!("{}: {e}", source.path.display()))?;
        Ok(0)
    }
}
