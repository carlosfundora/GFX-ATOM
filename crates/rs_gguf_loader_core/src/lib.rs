use std::fs::File;
use std::io::Read;
use std::path::Path;

use thiserror::Error;

pub mod profile;
pub mod llama_cpp_surface;
pub mod llama_cpp_family;

pub const GGUF_HEADER_BYTES: usize = 24;
const GGUF_MAGIC: [u8; 4] = *b"GGUF";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeaderV3 {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

impl GgufHeaderV3 {
    pub fn estimated_index_bytes(&self) -> u64 {
        // Conservative planning estimate for backend prefetch and pinned-ram staging.
        // Tensor table entries are variable-sized; 64 bytes each is a safe baseline envelope.
        (self.tensor_count.saturating_mul(64)) + (self.metadata_kv_count.saturating_mul(32))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufLoadPlan {
    pub prefetch_bytes: u64,
    pub io_chunk_bytes: u64,
    pub use_mmap: bool,
    pub use_pinned_staging: bool,
}

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("failed to read gguf file: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid gguf magic: {0:?}")]
    InvalidMagic([u8; 4]),
    #[error("unsupported gguf version: {0}")]
    UnsupportedVersion(u32),
    #[error("truncated gguf header: expected at least {expected} bytes, got {actual}")]
    TruncatedHeader { expected: usize, actual: usize },
}

pub fn parse_gguf_header_bytes(raw: &[u8]) -> Result<GgufHeaderV3, GgufError> {
    if raw.len() < GGUF_HEADER_BYTES {
        return Err(GgufError::TruncatedHeader {
            expected: GGUF_HEADER_BYTES,
            actual: raw.len(),
        });
    }

    let magic = [raw[0], raw[1], raw[2], raw[3]];
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic(magic));
    }

    let version = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    if version < 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let tensor_count = u64::from_le_bytes([
        raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14], raw[15],
    ]);
    let metadata_kv_count = u64::from_le_bytes([
        raw[16], raw[17], raw[18], raw[19], raw[20], raw[21], raw[22], raw[23],
    ]);

    Ok(GgufHeaderV3 {
        version,
        tensor_count,
        metadata_kv_count,
    })
}

pub fn parse_gguf_header_path(path: &Path) -> Result<GgufHeaderV3, GgufError> {
    let mut file = File::open(path)?;
    let mut header = [0u8; GGUF_HEADER_BYTES];
    file.read_exact(&mut header)?;
    parse_gguf_header_bytes(&header)
}

pub fn synthesize_load_plan(header: &GgufHeaderV3, max_prefetch_mb: u64) -> GgufLoadPlan {
    let max_prefetch_bytes = max_prefetch_mb.saturating_mul(1024 * 1024);
    let prefetch_bytes = header.estimated_index_bytes().min(max_prefetch_bytes);

    // Keep chunk sizes aligned for gfx1030-friendly IO cadence.
    let io_chunk_bytes = if prefetch_bytes <= 4 * 1024 * 1024 {
        1 * 1024 * 1024
    } else if prefetch_bytes <= 64 * 1024 * 1024 {
        4 * 1024 * 1024
    } else {
        8 * 1024 * 1024
    };

    GgufLoadPlan {
        prefetch_bytes,
        io_chunk_bytes,
        use_mmap: prefetch_bytes >= 8 * 1024 * 1024,
        use_pinned_staging: prefetch_bytes >= 2 * 1024 * 1024,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_gguf_header_bytes, synthesize_load_plan, GgufError, GgufHeaderV3,
        GGUF_HEADER_BYTES,
    };

    fn sample_header_bytes(version: u32, tensors: u64, kv: u64) -> [u8; GGUF_HEADER_BYTES] {
        let mut out = [0u8; GGUF_HEADER_BYTES];
        out[0..4].copy_from_slice(b"GGUF");
        out[4..8].copy_from_slice(&version.to_le_bytes());
        out[8..16].copy_from_slice(&tensors.to_le_bytes());
        out[16..24].copy_from_slice(&kv.to_le_bytes());
        out
    }

    #[test]
    fn parses_v3_header() {
        let raw = sample_header_bytes(3, 4096, 128);
        let header = parse_gguf_header_bytes(&raw).expect("header should parse");
        assert_eq!(
            header,
            GgufHeaderV3 {
                version: 3,
                tensor_count: 4096,
                metadata_kv_count: 128
            }
        );
    }

    #[test]
    fn rejects_invalid_magic() {
        let mut raw = sample_header_bytes(3, 1, 1);
        raw[0] = b'X';
        let err = parse_gguf_header_bytes(&raw).expect_err("should reject invalid magic");
        match err {
            GgufError::InvalidMagic(_) => {}
            _ => panic!("unexpected error variant: {err}"),
        }
    }

    #[test]
    fn rejects_unsupported_versions() {
        let raw = sample_header_bytes(2, 4, 2);
        let err = parse_gguf_header_bytes(&raw).expect_err("version 2 should be rejected");
        match err {
            GgufError::UnsupportedVersion(version) => assert_eq!(version, 2),
            _ => panic!("unexpected error variant: {err}"),
        }
    }

    #[test]
    fn load_plan_scales_prefetch_and_flags() {
        let header = GgufHeaderV3 {
            version: 3,
            tensor_count: 500_000,
            metadata_kv_count: 2_000,
        };
        let plan = synthesize_load_plan(&header, 64);
        assert_eq!(plan.prefetch_bytes, header.estimated_index_bytes());
        assert!(plan.use_mmap);
        assert!(plan.use_pinned_staging);
    }
}
