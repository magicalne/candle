use candle::quantized::gguf_file::TensorInfo;
use candle::quantized::QTensor;
use candle::{Device, Result, Shape};
use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

// VarBuilder specialized for QTensors
#[derive(Clone)]
pub struct VarBuilder {
    data: std::collections::HashMap<String, Arc<TensorInfo>>,
    path: Vec<String>,
    device: Device,
    file: Arc<File>,
    tensor_data_offset: u64,
}

impl VarBuilder {
    pub fn from_gguf<P: AsRef<std::path::Path>>(p: P, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(p)?;
        let content = candle::quantized::gguf_file::Content::read(&mut file)?;
        let data = content
            .tensor_infos
            .into_iter()
            .map(|ti| (ti.0, Arc::new(ti.1)))
            .collect();
        Ok(Self {
            file: Arc::new(file),
            data,
            path: Vec::new(),
            tensor_data_offset: content.tensor_data_offset,
            device: device.clone(),
        })
    }

    pub fn pp<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            file: self.file.clone(),
            data: self.data.clone(),
            path,
            tensor_data_offset: self.tensor_data_offset,
            device: self.device.clone(),
        }
    }

    fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        }
    }

    fn tensor(&mut self, tensor_info: Arc<TensorInfo>) -> Result<QTensor> {
        tensor_info.read(&mut self.file, self.tensor_data_offset, &self.device)
    }

    pub fn get<S: Into<Shape>>(&mut self, s: S, name: &str) -> Result<QTensor> {
        let path = self.path(name);
        match self.data.get(&path) {
            None => {
                candle::bail!("cannot find tensor {path}")
            }
            Some(tensor_info) => {
                let shape = s.into();
                if tensor_info.shape != shape {
                    candle::bail!(
                        "shape mismatch for {name}, got {:?}, expected {shape:?}",
                        tensor_info.shape
                    )
                }
                let instant = Instant::now();
                let qtensor = self.tensor(tensor_info.clone())?;
                info!(
                    "load tensor: {} costs: {}s",
                    name,
                    instant.elapsed().as_secs()
                );
                Ok(qtensor)
            }
        }
    }

    pub fn get_no_shape(&mut self, name: &str) -> Result<QTensor> {
        let path = self.path(name);
        match self.data.get(&path) {
            None => {
                candle::bail!("cannot find tensor {name}")
            }
            Some(tensor_info) => self.tensor(tensor_info.clone()),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    pub fn tensor_infos(&self) -> &HashMap<String, Arc<TensorInfo>> {
        &self.data
    }
}
