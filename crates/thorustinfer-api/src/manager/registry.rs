use super::{Result, WorkerOptions};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
};
use thorustinfer_engine::artifact::sha256;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRecord {
    pub name: String,
    pub path: PathBuf,
    pub identity: String,
    pub architecture: String,
    pub execution_package: String,
    pub weight_bytes: usize,
    pub load_bytes: usize,
    #[serde(default)]
    pub max_context: usize,
    #[serde(default)]
    pub sequence_bytes: usize,
    #[serde(default)]
    pub kv_bytes_per_token: usize,
    pub options: WorkerOptions,
}

impl ModelRecord {
    pub(super) fn start_bytes(&self) -> Result<usize> {
        self.load_bytes
            .checked_add(self.options.context_options.kv_bytes()?)
            .ok_or("Model load budget overflow".into())
    }
}

pub(super) fn model_identity(target: &[u8], options: &WorkerOptions) -> Result<String> {
    let id = sha256(target);
    let Some(path) = &options.draft_model else {
        return Ok(id);
    };
    let descriptor =
        if options.spec_method == Some(thorustinfer_engine::execution::SpecMethod::Draft) {
            "cache/model.json"
        } else {
            "draft.json"
        };
    let data = fs::read(path.join(descriptor)).map_err(|e| e.to_string())?;
    Ok(sha256(format!("{id}:{}", sha256(&data)).as_bytes()))
}

pub fn inspect(name: &str, path: &Path, options: WorkerOptions) -> Result<ModelRecord> {
    validate_name(name)?;
    let path = path.canonicalize().map_err(|e| e.to_string())?;
    options.settings(name, &path)?.validate()?;
    let plan = thorustinfer_engine::model::inspect_plan(&path)?;
    let raw = fs::read(path.join("cache/model.json")).map_err(|e| e.to_string())?;
    let raw_descriptor: serde_json::Value =
        serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    let mut load_bytes = 512usize << 20;
    let mut sequence_bytes = 0usize;
    for buffer in &plan.manifest.buffers {
        let demand = plan.manifest.kv_cache.as_ref().is_some_and(|kv| {
            kv.demand_mapping
                && (kv.buffers.contains_key(&buffer.name)
                    || kv.prefill_workspace.contains_key(&buffer.name))
        });
        if !demand {
            // Cold loading creates an image-free bootstrap sequence. The GPU
            // worker admits the actual request's feature bank separately.
            let bytes = plan
                .manifest
                .vision
                .as_ref()
                .filter(|v| {
                    v.features == buffer.name
                        && raw_descriptor["buffer_scopes"][&buffer.name] == "sequence"
                })
                .map_or_else(|| buffer.bytes(), |v| Ok(v.hidden * 2))?;
            load_bytes = load_bytes
                .checked_add(bytes)
                .ok_or("Model memory estimate overflow")?;
            if raw_descriptor["buffer_scopes"][&buffer.name] == "sequence" {
                sequence_bytes = sequence_bytes
                    .checked_add(bytes)
                    .ok_or("Sequence memory estimate overflow")?;
            }
        }
    }
    let mut kv_bytes_per_token = plan
        .manifest
        .kv_cache
        .as_ref()
        .map(|kv| {
            kv.buffers.values().try_fold(0usize, |n, value| {
                n.checked_add(*value).ok_or("KV memory estimate overflow")
            })
        })
        .transpose()?
        .unwrap_or(0);
    let mut max_context = plan.manifest.max_context;
    let identity = model_identity(&raw, &options)?;
    if let Some(draft) = &options.draft_model {
        let requested =
            if options.spec_method == Some(thorustinfer_engine::execution::SpecMethod::Draft) {
                let other = inspect(name, draft, WorkerOptions::default())?;
                max_context = max_context.min(other.max_context);
                sequence_bytes = sequence_bytes.saturating_add(other.sequence_bytes);
                kv_bytes_per_token = kv_bytes_per_token.saturating_add(other.kv_bytes_per_token);
                other.load_bytes
            } else {
                let bytes = std::fs::metadata(draft.join("model.safetensors"))
                    .map_err(|e| e.to_string())?
                    .len();
                usize::try_from(bytes).map_err(|_| "Draft weights exceed addressable memory")?
            };
        load_bytes = load_bytes
            .checked_add(requested)
            .ok_or("Draft load budget overflow")?;
    }
    if let Some(thorustinfer_engine::execution::ContextLimit::Tokens(n)) =
        options.context_options.context
    {
        if n.get() > max_context {
            return Err(format!(
                "Requested context exceeds package limit {max_context}"
            ));
        }
        max_context = n.get();
    }
    Ok(ModelRecord {
        name: name.into(),
        path,
        identity,
        architecture: plan.architecture.to_string(),
        execution_package: plan.execution_package,
        weight_bytes: plan.manifest.weight_bytes,
        load_bytes,
        max_context,
        sequence_bytes,
        kv_bytes_per_token,
        options,
    })
}

pub fn validate_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 128
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_:./".contains(&b))
        || name.starts_with('/')
        || name
            .split('/')
            .any(|s| s.is_empty() || s == "." || s == "..")
    {
        return Err(
            "Model name must be 1..128 letters, digits or -_:./ without relative path segments"
                .into(),
        );
    }
    Ok(())
}

pub fn read(root: &Path) -> Result<BTreeMap<String, ModelRecord>> {
    let path = root.join("models.json");
    match fs::read(&path) {
        Ok(raw) => serde_json::from_slice(&raw).map_err(|e| format!("Model registry: {e}")),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(e) => Err(e.to_string()),
    }
}

pub fn write(root: &Path, records: &BTreeMap<String, ModelRecord>) -> Result<()> {
    let temporary = root.join("models.json.next");
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| e.to_string())?;
        file.write_all(&serde_json::to_vec_pretty(records).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        fs::rename(&temporary, root.join("models.json")).map_err(|e| e.to_string())?;
        fs::File::open(root)
            .and_then(|d| d.sync_all())
            .map_err(|e| e.to_string())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aliases_cannot_escape_the_registry() {
        for name in ["index-translate:2b", "owner/model:q4"] {
            validate_name(name).unwrap();
        }
        for name in ["", "../x", "/x", "a//b", "a/../b", "hello world"] {
            assert!(validate_name(name).is_err());
        }
    }
}
