use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Zero dimension not allowed")]
    ZeroDimension,

    #[error("Odd dimension {got} not allowed (must be even for polar pairing)")]
    OddDimension { got: usize },

    #[error("Invalid bit width {got}: must be 1-8")]
    InvalidBitWidth { got: u8 },

    #[error("Zero projection count not allowed")]
    ZeroProjectionCount,

    #[error("Compression error: {0}")]
    CompressionError(String),

    #[error("Decompression error: {0}")]
    DecompressionError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}
