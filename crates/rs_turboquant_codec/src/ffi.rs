//! FFI bindings for PyO3 Python integration
#![cfg(feature = "python")]

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray4};
use ndarray::{Array1, Array4, ArrayView1, ArrayView4};

use crate::polar::PolarQuantizer;
use crate::qjl::QJLQuantizer;
use crate::turbo::TurboQuantizer;

/// Python wrapper for TurboQuantizer codec
#[pyclass]
pub struct PyTurboQuantCodec {
    mode: String,
    qjl_projections: usize,
}

#[pymethods]
impl PyTurboQuantCodec {
    /// Create a new TurboQuantizer instance
    /// 
    /// Args:
    ///     mode: Compression mode ("tq1", "tq2", "tq3", "tq4")
    ///     qjl_projections: Number of JL projections (default: 256)
    #[new]
    pub fn new(mode: String, qjl_projections: Option<usize>) -> PyResult<Self> {
        // Validate mode
        match mode.as_str() {
            "tq1" | "tq2" | "tq3" | "tq4" => {},
            _ => return Err(PyValueError::new_err(
                format!("Invalid mode: {}. Must be tq1, tq2, tq3, or tq4", mode)
            )),
        }
        
        Ok(PyTurboQuantCodec {
            mode,
            qjl_projections: qjl_projections.unwrap_or(256),
        })
    }
    
    /// Get the compression mode
    #[getter]
    pub fn get_mode(&self) -> String {
        self.mode.clone()
    }
    
    /// Encode KV cache with TurboQuantizer
    /// 
    /// Args:
    ///     kv: 4D numpy array (batch, seq_len, heads, dim) of float32
    ///     seed: Random seed for reproducibility
    /// 
    /// Returns:
    ///     1D numpy array of compressed data as uint8
    pub fn encode<'py>(
        &self,
        py: Python<'py>,
        kv: &PyArray4<f32>,
        seed: Option<u64>,
    ) -> PyResult<&'py PyArray1<u8>> {
        let seed = seed.unwrap_or(42);
        let kv_array = kv.as_array();
        
        match self.mode.as_str() {
            "tq1" => self._encode_tq1(py, &kv_array, seed),
            "tq2" => self._encode_tq2(py, &kv_array, seed),
            "tq3" => self._encode_tq3(py, &kv_array, seed),
            "tq4" => self._encode_tq4(py, &kv_array, seed),
            _ => unreachable!(),
        }
    }
    
    /// Decode compressed KV cache
    /// 
    /// Args:
    ///     compressed: 1D numpy array of compressed data
    ///     shape: Tuple (batch, seq_len, heads, dim)
    ///     seed: Random seed (must match encode)
    /// 
    /// Returns:
    ///     4D numpy array of decompressed float32 data
    pub fn decode<'py>(
        &self,
        py: Python<'py>,
        compressed: &PyArray1<u8>,
        shape: (usize, usize, usize, usize),
        seed: Option<u64>,
    ) -> PyResult<&'py PyArray4<f32>> {
        let seed = seed.unwrap_or(42);
        let compressed_data = compressed.to_vec()?;
        
        match self.mode.as_str() {
            "tq1" => self._decode_tq1(py, &compressed_data, shape, seed),
            "tq2" => self._decode_tq2(py, &compressed_data, shape, seed),
            "tq3" => self._decode_tq3(py, &compressed_data, shape, seed),
            "tq4" => self._decode_tq4(py, &compressed_data, shape, seed),
            _ => unreachable!(),
        }
    }
    
    /// Estimate inner product between two KV caches (TQ mode)
    /// 
    /// Args:
    ///     kv1: First 4D array
    ///     kv2: Second 4D array
    /// 
    /// Returns:
    ///     Estimated inner product (float32)
    pub fn estimate_inner_product(
        &self,
        kv1: &PyArray4<f32>,
        kv2: &PyArray4<f32>,
    ) -> PyResult<f32> {
        let kv1_array = kv1.as_array();
        let kv2_array = kv2.as_array();
        
        if kv1_array.shape() != kv2_array.shape() {
            return Err(PyValueError::new_err("KV shapes must match"));
        }
        
        let ip = crate::turbo::estimate_inner_product_batch(&kv1_array, &kv2_array);
        Ok(ip)
    }
    
    /// Get compression ratio metadata
    pub fn compression_ratio(&self) -> f32 {
        match self.mode.as_str() {
            "tq1" => 16.0,  // 32 bits -> 2 bits per float
            "tq2" => 8.0,   // 32 bits -> 4 bits per float
            "tq3" => 5.3,   // 32 bits -> 6 bits per float
            "tq4" => 4.0,   // 32 bits -> 8 bits per float
            _ => 1.0,
        }
    }
}

impl PyTurboQuantCodec {
    fn _encode_tq1<'py>(
        &self,
        py: Python<'py>,
        kv: &ArrayView4<f32>,
        seed: u64,
    ) -> PyResult<&'py PyArray1<u8>> {
        let turbo = TurboQuantizer::new(1, seed);
        let encoded = turbo.encode(kv).map_err(|e| {
            PyValueError::new_err(format!("Encode failed: {}", e))
        })?;
        
        Ok(PyArray1::from_vec(py, encoded))
    }
    
    fn _encode_tq2<'py>(
        &self,
        py: Python<'py>,
        kv: &ArrayView4<f32>,
        seed: u64,
    ) -> PyResult<&'py PyArray1<u8>> {
        let turbo = TurboQuantizer::new(2, seed);
        let encoded = turbo.encode(kv).map_err(|e| {
            PyValueError::new_err(format!("Encode failed: {}", e))
        })?;
        
        Ok(PyArray1::from_vec(py, encoded))
    }
    
    fn _encode_tq3<'py>(
        &self,
        py: Python<'py>,
        kv: &ArrayView4<f32>,
        seed: u64,
    ) -> PyResult<&'py PyArray1<u8>> {
        let turbo = TurboQuantizer::new(3, seed);
        let encoded = turbo.encode(kv).map_err(|e| {
            PyValueError::new_err(format!("Encode failed: {}", e))
        })?;
        
        Ok(PyArray1::from_vec(py, encoded))
    }
    
    fn _encode_tq4<'py>(
        &self,
        py: Python<'py>,
        kv: &ArrayView4<f32>,
        seed: u64,
    ) -> PyResult<&'py PyArray1<u8>> {
        let turbo = TurboQuantizer::new(4, seed);
        let encoded = turbo.encode(kv).map_err(|e| {
            PyValueError::new_err(format!("Encode failed: {}", e))
        })?;
        
        Ok(PyArray1::from_vec(py, encoded))
    }
    
    fn _decode_tq1<'py>(
        &self,
        py: Python<'py>,
        compressed: &[u8],
        shape: (usize, usize, usize, usize),
        seed: u64,
    ) -> PyResult<&'py PyArray4<f32>> {
        let turbo = TurboQuantizer::new(1, seed);
        let decoded = turbo.decode(compressed, shape).map_err(|e| {
            PyValueError::new_err(format!("Decode failed: {}", e))
        })?;
        
        let arr = Array4::from_shape_vec(shape, decoded.to_vec()).map_err(|e| {
            PyValueError::new_err(format!("Shape mismatch: {}", e))
        })?;
        
        Ok(PyArray4::from_array(py, &arr))
    }
    
    fn _decode_tq2<'py>(
        &self,
        py: Python<'py>,
        compressed: &[u8],
        shape: (usize, usize, usize, usize),
        seed: u64,
    ) -> PyResult<&'py PyArray4<f32>> {
        let turbo = TurboQuantizer::new(2, seed);
        let decoded = turbo.decode(compressed, shape).map_err(|e| {
            PyValueError::new_err(format!("Decode failed: {}", e))
        })?;
        
        let arr = Array4::from_shape_vec(shape, decoded.to_vec()).map_err(|e| {
            PyValueError::new_err(format!("Shape mismatch: {}", e))
        })?;
        
        Ok(PyArray4::from_array(py, &arr))
    }
    
    fn _decode_tq3<'py>(
        &self,
        py: Python<'py>,
        compressed: &[u8],
        shape: (usize, usize, usize, usize),
        seed: u64,
    ) -> PyResult<&'py PyArray4<f32>> {
        let turbo = TurboQuantizer::new(3, seed);
        let decoded = turbo.decode(compressed, shape).map_err(|e| {
            PyValueError::new_err(format!("Decode failed: {}", e))
        })?;
        
        let arr = Array4::from_shape_vec(shape, decoded.to_vec()).map_err(|e| {
            PyValueError::new_err(format!("Shape mismatch: {}", e))
        })?;
        
        Ok(PyArray4::from_array(py, &arr))
    }
    
    fn _decode_tq4<'py>(
        &self,
        py: Python<'py>,
        compressed: &[u8],
        shape: (usize, usize, usize, usize),
        seed: u64,
    ) -> PyResult<&'py PyArray4<f32>> {
        let turbo = TurboQuantizer::new(4, seed);
        let decoded = turbo.decode(compressed, shape).map_err(|e| {
            PyValueError::new_err(format!("Decode failed: {}", e))
        })?;
        
        let arr = Array4::from_shape_vec(shape, decoded.to_vec()).map_err(|e| {
            PyValueError::new_err(format!("Shape mismatch: {}", e))
        })?;
        
        Ok(PyArray4::from_array(py, &arr))
    }
}

/// Python module for turboquant codec
#[pymodule]
pub fn turboquant(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTurboQuantCodec>()?;
    Ok(())
}
