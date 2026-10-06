use anyhow::{Context, Result};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeightInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

pub struct WeightsManager;

impl WeightsManager {
    pub fn inspect_checkpoint<P: AsRef<Path>>(path: P) -> Result<Vec<ModelWeightInfo>> {
        let mut file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open checkpoint {:?}", path.as_ref()))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let tensors = SafeTensors::deserialize(&buffer)
            .with_context(|| "Failed to parse SafeTensors checkpoint")?;

        let mut infos = Vec::new();
        for (name, view) in tensors.tensors() {
            infos.push(ModelWeightInfo {
                name,
                shape: view.shape().to_vec(),
                dtype: format!("{:?}", view.dtype()),
            });
        }
        infos.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(infos)
    }
}
