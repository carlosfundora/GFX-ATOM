use crate::error::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// QJL sketch (1-bit residual per projection)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct QjlSketch {
    pub projections: usize,
    pub bits: Vec<u8>,
}

impl QjlSketch {
    pub fn encoded_bytes(&self) -> usize {
        self.bits.len()
    }
}

/// QjlQuantizer: Second stage of TurboQuant
#[derive(Debug, Clone)]
pub struct QjlQuantizer {
    projections: usize,
    seed: u64,
}

impl QjlQuantizer {
    pub fn new(projections: usize, seed: u64) -> Result<Self> {
        if projections == 0 {
            return Err(crate::Error::ZeroProjectionCount);
        }
        Ok(Self { projections, seed })
    }

    pub fn encode(&self, _x: &[f32]) -> Result<QjlSketch> {
        let encoded_bytes = (self.projections + 7) / 8;
        Ok(QjlSketch {
            projections: self.projections,
            bits: vec![0u8; encoded_bytes],
        })
    }

    pub fn decode(&self, _sketch: &QjlSketch) -> Result<Vec<f32>> {
        Ok(vec![0.0; self.projections])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qjl_quantizer_creation() {
        let qjl = QjlQuantizer::new(32, 42).unwrap();
        assert_eq!(qjl.projections, 32);
    }

    #[test]
    fn test_qjl_encode() {
        let qjl = QjlQuantizer::new(16, 42).unwrap();
        let x = vec![0.5; 64];
        let sketch = qjl.encode(&x).unwrap();
        assert_eq!(sketch.projections, 16);
    }
}
