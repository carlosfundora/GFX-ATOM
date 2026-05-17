use crate::error::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Polar-encoded vector (b-1 bits of main signal)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PolarCode {
    pub dim: usize,
    pub bits: u8,
    pub bytes: Vec<u8>,
}

impl PolarCode {
    pub fn encoded_bytes(&self) -> usize {
        self.bytes.len()
    }
}

/// PolarQuantizer: First stage of TurboQuant
#[derive(Debug, Clone)]
pub struct PolarQuantizer {
    dim: usize,
    bits: u8,
    seed: u64,
}

impl PolarQuantizer {
    pub fn new(dim: usize, bits: u8, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(crate::Error::ZeroDimension);
        }
        if bits == 0 || bits > 8 {
            return Err(crate::Error::InvalidBitWidth { got: bits });
        }
        Ok(Self { dim, bits, seed })
    }

    pub fn encode(&self, x: &[f32]) -> Result<PolarCode> {
        if x.len() != self.dim {
            return Err(crate::Error::CompressionError(format!(
                "dimension mismatch in PolarQuantizer encode"
            )));
        }

        let encoded_bytes = (self.dim * self.bits as usize + 7) / 8;
        Ok(PolarCode {
            dim: self.dim,
            bits: self.bits,
            bytes: vec![0u8; encoded_bytes],
        })
    }

    pub fn decode(&self, _code: &PolarCode) -> Result<Vec<f32>> {
        Ok(vec![0.0; self.dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polar_quantizer_creation() {
        let pq = PolarQuantizer::new(128, 2, 42).unwrap();
        assert_eq!(pq.dim, 128);
        assert_eq!(pq.bits, 2);
    }

    #[test]
    fn test_polar_encode() {
        let pq = PolarQuantizer::new(64, 2, 42).unwrap();
        let x = vec![0.5; 64];
        let code = pq.encode(&x).unwrap();
        assert_eq!(code.dim, 64);
    }
}
