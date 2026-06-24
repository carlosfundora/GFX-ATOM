use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyBytes;
use numpy::ndarray::Array1;
use rayon::prelude::*;
use base64::{Engine as _, engine::general_purpose};
use blake2::{Blake2b, Digest};
use blake2::digest::consts::U8;
use std::collections::HashMap;

type Blake2b64 = Blake2b<U8>;

#[pyfunction]
fn decode_raw_f32(
    py: Python,
    audio_bytes: &[u8],
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    let mut out: Array1<f32>;

    if dtype == "float32" {
        // Ensure multiple of 4 bytes
        let valid_len = (audio_bytes.len() / 4) * 4;
        let slice = &audio_bytes[..valid_len];

        let floats = bytemuck::cast_slice::<u8, f32>(slice);
        out = Array1::from_vec(floats.to_vec());
    } else {
        // Fallback to int16 (multiple of 2)
        let valid_len = (audio_bytes.len() / 2) * 2;
        let slice = &audio_bytes[..valid_len];

        let ints = bytemuck::cast_slice::<u8, i16>(slice);
        out = Array1::<f32>::zeros(ints.len());
        for i in 0..ints.len() {
            out[i] = (ints[i] as f32) / 32768.0;
        }
    }

    Ok(out.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
fn decode_b64_f32(
    py: Python,
    audio_b64: &str,
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    let decoded_bytes = match general_purpose::STANDARD.decode(audio_b64) {
        Ok(b) => b,
        Err(_) => return Err(pyo3::exceptions::PyValueError::new_err("Invalid base64")),
    };
    decode_raw_f32(py, &decoded_bytes, dtype)
}

#[pyfunction]
fn encode_wav_b64(
    audio: PyReadonlyArray1<f32>,
    sample_rate: u32,
    _subtype: &str,
) -> PyResult<String> {
    let audio_view = audio.as_array();
    let num_samples = audio_view.len();
    let num_channels: u32 = 1;
    let bytes_per_sample: u32 = 2;
    let byte_rate = sample_rate * num_channels * bytes_per_sample;
    let data_chunk_size = num_samples as u32 * num_channels * bytes_per_sample;
    let riff_chunk_size = 36 + data_chunk_size;

    let mut out = Vec::with_capacity((44 + data_chunk_size) as usize);

    // RIFF chunk descriptor
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_chunk_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // "fmt " sub-chunk
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&(num_channels as u16).to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&((num_channels * bytes_per_sample) as u16).to_le_bytes());
    out.extend_from_slice(&((bytes_per_sample * 8) as u16).to_le_bytes());

    // "data" sub-chunk
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_chunk_size.to_le_bytes());

    for &x in audio_view.iter() {
        let clamped = (x * 32768.0).clamp(-32768.0, 32767.0) as i16;
        out.extend_from_slice(&clamped.to_le_bytes());
    }

    Ok(general_purpose::STANDARD.encode(&out))
}

#[pyfunction]
fn crossfade_chunks(
    py: Python,
    prev_tail: PyReadonlyArray1<f32>,
    next_head: PyReadonlyArray1<f32>,
    fade_samples: usize,
) -> Py<PyAny> {
    let prev = prev_tail.as_array();
    let next = next_head.as_array();
    let n = fade_samples.min(prev.len()).min(next.len());

    if n < 2 {
        let val = if next.len() >= n && n > 0 { next[0] } else if prev.len() > 0 { prev[prev.len() - 1] } else { 0.0 };
        let out = Array1::from_vec(vec![val; n.max(1).min(1)]);
        return out.into_pyarray(py).into_any().unbind();
    }

    let mut out = Array1::<f32>::zeros(n);
    let half_pi = std::f32::consts::FRAC_PI_2;
    let prev_offset = prev.len() - n;

    for i in 0..n {
        let t = (i as f32) / ((n - 1) as f32) * half_pi;
        let fade_out = t.cos();
        let fade_in = t.sin();
        out[i] = prev[prev_offset + i] * fade_out + next[i] * fade_in;
    }

    out.into_pyarray(py).into_any().unbind()
}

#[pyfunction]
fn float_to_int16_bytes(
    py: Python,
    frame: PyReadonlyArray1<f32>,
) -> PyResult<PyObject> {
    let frame_view = frame.as_array();
    let mut ints = Vec::with_capacity(frame_view.len());

    for &x in frame_view.iter() {
        let scaled = x * 32768.0;
        let clamped = scaled.clamp(-32768.0, 32767.0);
        ints.push(clamped as i16);
    }

    let bytes_slice = bytemuck::cast_slice::<i16, u8>(&ints);
    let py_bytes = PyBytes::new(py, bytes_slice);
    Ok(py_bytes.into())
}

#[pyfunction]
fn soft_compressor(
    py: Python,
    audio: PyReadonlyArray1<f32>,
    threshold: f32,
    ratio: f32,
    attack: f32,
    release: f32,
    initial_gain: f32,
) -> (Py<PyAny>, f32) {
    let audio_view = audio.as_array();
    let mut out = Array1::<f32>::zeros(audio_view.len());
    let mut gain = initial_gain;

    for (i, &x) in audio_view.iter().enumerate() {
        let ax = x.abs();
        if ax > threshold {
            let target = threshold + (ax - threshold) / ratio;
            gain = gain * attack + (target / (ax + 1e-8)) * (1.0 - attack);
        } else {
            gain = gain * release + 1.0 * (1.0 - release);
        }
        out[i] = x * gain;
    }

    (out.into_pyarray(py).into_any().unbind(), gain)
}

#[pyfunction]
fn agc_kernel(
    py: Python,
    audio: PyReadonlyArray1<f32>,
    target_rms: f32,
    attack: f32,
    release: f32,
    max_gain: f32,
    window: usize,
    initial_gain: f32,
) -> (Py<PyAny>, f32) {
    let audio_view = audio.as_array();
    let mut out = Array1::<f32>::zeros(audio_view.len());
    let mut current_gain = initial_gain;
    let mut s: f32 = 0.0;

    for (i, &x) in audio_view.iter().enumerate() {
        s += x * x;
        if i >= window {
            let old_x = audio_view[i - window];
            s -= old_x * old_x;
        }
        if s < 0.0 {
            s = 0.0;
        }
        let n_samples = if i < window { (i + 1) as f32 } else { window as f32 };
        let rms = (s / n_samples).sqrt() + 1e-8;
        let desired_gain = (target_rms / rms).min(max_gain);

        if desired_gain > current_gain {
            current_gain = current_gain * attack + desired_gain * (1.0 - attack);
        } else {
            current_gain = current_gain * release + desired_gain * (1.0 - release);
        }
        out[i] = x * current_gain;
    }

    (out.into_pyarray(py).into_any().unbind(), current_gain)
}

#[pyfunction]
fn iir_1pole_kernel(
    py: Python,
    audio: PyReadonlyArray1<f32>,
    a: f32,
    b: f32,
    initial_y: f32,
) -> (Py<PyAny>, f32) {
    let audio_view = audio.as_array();
    let mut out = Array1::<f32>::zeros(audio_view.len());
    if audio_view.is_empty() {
        return (out.into_pyarray(py).into_any().unbind(), initial_y);
    }

    out[0] = b * audio_view[0] + a * initial_y;
    for i in 1..audio_view.len() {
        out[i] = b * audio_view[i] + a * out[i - 1];
    }

    let last_y = out[out.len() - 1];
    (out.into_pyarray(py).into_any().unbind(), last_y)
}

#[pyfunction]
fn highpass_kernel(
    py: Python,
    audio: PyReadonlyArray1<f32>,
    a: f32,
    b: f32,
    initial_y: f32,
    initial_x: f32,
) -> (Py<PyAny>, f32, f32) {
    let audio_view = audio.as_array();
    let mut out = Array1::<f32>::zeros(audio_view.len());

    if audio_view.is_empty() {
        return (out.into_pyarray(py).into_any().unbind(), initial_y, initial_x);
    }

    out[0] = b * (audio_view[0] - initial_x) + a * initial_y;
    for i in 1..audio_view.len() {
        out[i] = b * (audio_view[i] - audio_view[i - 1]) + a * out[i - 1];
    }

    let last_y = out[out.len() - 1];
    let last_x = audio_view[audio_view.len() - 1];
    (out.into_pyarray(py).into_any().unbind(), last_y, last_x)
}

#[pyfunction]
fn fused_preprocess_for_vad_f32(
    py: Python,
    audio: PyReadonlyArray1<f32>,
    sr: f32,
    hp_y: f32,
    hp_x: f32,
    mut noise_floor_db: f32,
    mut agc_gain: f32,
    hp_cutoff_hz: f32,
    frame_size: usize,
    noise_alpha: f32,
    gate_margin_db: f32,
    gate_floor_gain: f32,
    target_rms: f32,
    agc_attack: f32,
    agc_release: f32,
    max_gain: f32,
    limiter_ceiling: f32,
) -> (Py<PyAny>, f32, f32, f32, f32) {
    let audio_view = audio.as_array();
    let mut out = Array1::<f32>::zeros(audio_view.len());

    if audio_view.is_empty() {
        return (out.into_pyarray(py).into_any().unbind(), hp_y, hp_x, noise_floor_db, agc_gain);
    }

    let omega = 2.0 * std::f32::consts::PI * hp_cutoff_hz / sr;
    let sin_omega = omega.sin();
    let cos_omega = omega.cos();
    let alpha = sin_omega / 2.0;
    
    let hp_b0 = (1.0 + cos_omega) / 2.0;
    let hp_b1 = -(1.0 + cos_omega);
    let hp_a0 = 1.0 + alpha;
    let hp_a1 = -2.0 * cos_omega;
    let hp_a2 = 1.0 - alpha;

    let hp_b0_norm = hp_b0 / hp_a0;
    let hp_b1_norm = hp_b1 / hp_a0;
    let hp_a1_norm = hp_a1 / hp_a0;
    let hp_a2_norm = hp_a2 / hp_a0;

    // For noise floor updates
    let mut rms_sum: f32 = 0.0;
    let mut hp_y1: f32 = hp_y;
    let mut hp_y2: f32 = hp_x;

    for (i, &x) in audio_view.iter().enumerate() {
        // Highpass filter: 2nd-order IIR
        let hp_out = hp_b0_norm * x + hp_b1_norm * hp_y 
            - hp_a1_norm * hp_y1 - hp_a2_norm * hp_y2;
        
        hp_y2 = hp_y1;
        hp_y1 = hp_out;

        // RMS calculation for noise gate
        let frame_idx = i % frame_size;
        rms_sum += hp_out * hp_out;

        // Noise floor update
        if (i + 1) % frame_size == 0 {
            let frame_rms = (rms_sum / (frame_size as f32)).sqrt() + 1e-8;
            let frame_db = 20.0 * frame_rms.log10();
            noise_floor_db = noise_floor_db * (1.0 - noise_alpha) + frame_db * noise_alpha;
            rms_sum = 0.0;
        }

        // Gate logic
        let gate_threshold_db = noise_floor_db + gate_margin_db;
        let frame_rms_current = (rms_sum / ((frame_idx + 1) as f32)).sqrt() + 1e-8;
        let frame_db_current = 20.0 * frame_rms_current.log10();
        
        let gate_gain = if frame_db_current < gate_threshold_db {
            gate_floor_gain
        } else {
            1.0
        };

        // AGC
        let rms_for_agc = (hp_out * hp_out + 1e-8).sqrt();
        let desired_gain = (target_rms / (rms_for_agc + 1e-8)).min(max_gain);
        
        agc_gain = if desired_gain > agc_gain {
            agc_gain * agc_attack + desired_gain * (1.0 - agc_attack)
        } else {
            agc_gain * agc_release + desired_gain * (1.0 - agc_release)
        };

        let gated = hp_out * gate_gain;
        let agc_out = gated * agc_gain;

        // Tanh limiter
        let limited = agc_out.tanh() * limiter_ceiling;

        out[i] = limited;
    }

    (out.into_pyarray(py).into_any().unbind(), hp_y1, hp_y2, noise_floor_db, agc_gain)
    let view = audio.as_array();
    let mut pcm_data = Vec::with_capacity(view.len() * 2);
    for &x in view.iter() {
        let val = (x * 32767.0).clamp(-32768.0, 32767.0) as i16;
        pcm_data.extend_from_slice(&val.to_le_bytes());
    }
    pyo3::types::PyBytes::new(py, &pcm_data)
        .into_any()
        .unbind()

#[pyfunction]
fn compute_rms(audio: PyReadonlyArray1<f32>) -> f64 {
    let audio_view = audio.as_array();
    if audio_view.is_empty() {
        return 0.0;
    }

    if let Some(slice) = audio_view.as_slice() {
        let sum_sq: f64 = slice.par_iter().map(|&x| (x as f64) * (x as f64)).sum();
        (((sum_sq) / (slice.len() as f64)) + 1e-12).sqrt()
    } else {
        let mut sum_sq: f64 = 0.0;
        for &x in audio_view.iter() {
            let x_f64 = x as f64;
            sum_sq += x_f64 * x_f64;
        }
        return ((sum_sq / (audio_view.len() as f64)) + 1e-12).sqrt();
    }
}

// ---------------------------------------------------------------------------
// SentenceSplitter — streaming sentence boundary detector (from gfxATOM)
// ---------------------------------------------------------------------------

#[pyclass]
pub struct SentenceSplitter {
    buffer: String,
    min_sentence_length: usize,
}

#[pymethods]
impl SentenceSplitter {
    #[new]
    #[pyo3(signature = (min_sentence_length=2))]
    fn new(min_sentence_length: usize) -> Self {
        SentenceSplitter {
            buffer: String::new(),
            min_sentence_length,
        }
    }

    #[getter]
    fn buffer(&self) -> String {
        self.buffer.clone()
    }

    fn add_text(&mut self, text: &str) -> PyResult<Vec<String>> {
        self.buffer.push_str(text);
        if self.buffer.len() > 100_000 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Text buffer exceeded maximum size (100,000 chars). Consider adding sentence-ending punctuation."
            ));
        }
        Ok(self.extract_sentences())
    }

    fn flush(&mut self) -> Option<String> {
        let remaining = self.buffer.trim().to_string();
        self.buffer.clear();
        if remaining.is_empty() {
            None
        } else {
            Some(remaining)
        }
    }
}

impl SentenceSplitter {
    fn extract_sentences(&mut self) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut carry = String::new();

        loop {
            let mut split_idx = None;
            let mut chars = self.buffer.char_indices().peekable();
            let mut boundary_end_idx = 0;

            while let Some((i, c)) = chars.next() {
                if c == '.' || c == '!' || c == '?' {
                    if let Some(&(next_i, next_c)) = chars.peek() {
                        if next_c.is_whitespace() {
                            let mut end = next_i;
                            while let Some(&(ws_i, ws_c)) = chars.peek() {
                                if ws_c.is_whitespace() {
                                    end = ws_i + ws_c.len_utf8();
                                    chars.next();
                                } else {
                                    break;
                                }
                            }
                            split_idx = Some(i + c.len_utf8());
                            boundary_end_idx = end;
                            break;
                        }
                    }
                } else if c == '\u{3002}' || c == '\u{FF01}' || c == '\u{FF1F}' {
                    // CJK sentence-ending punctuation: 。！？
                    split_idx = Some(i + c.len_utf8());
                    boundary_end_idx = i + c.len_utf8();
                    break;
                }
            }

            match split_idx {
                Some(idx) => {
                    let sentence = &self.buffer[..idx];
                    let text = format!("{}{}", carry, sentence);
                    carry.clear();

                    let stripped = text.trim();
                    if stripped.chars().count() >= self.min_sentence_length {
                        sentences.push(stripped.to_string());
                    } else if !stripped.is_empty() {
                        carry = text;
                    }

                    let remaining = self.buffer[boundary_end_idx..].to_string();
                    self.buffer = remaining;
                }
                None => {
                    if !carry.is_empty() {
                        self.buffer = format!("{}{}", carry, self.buffer);
                    }
                    break;
                }
            }
        }

        sentences
    }
}

// ---------------------------------------------------------------------------
// Hamming distance — Q1 sign-bit blocks with AVX2 dispatch (from rs_quant_kernels)
// ---------------------------------------------------------------------------

fn hamming_q1_blocks_scalar(lhs_bits: &[u8], rhs_bits: &[u8]) -> u32 {
    let lhs_chunks = lhs_bits.chunks_exact(8);
    let rhs_chunks = rhs_bits.chunks_exact(8);

    let mut sum: u32 = lhs_chunks
        .clone()
        .zip(rhs_chunks.clone())
        .map(|(l, r)| {
            let lx = u64::from_ne_bytes(l.try_into().expect("chunk size is 8"));
            let rx = u64::from_ne_bytes(r.try_into().expect("chunk size is 8"));
            (lx ^ rx).count_ones()
        })
        .sum();

    sum += lhs_chunks
        .remainder()
        .iter()
        .zip(rhs_chunks.remainder().iter())
        .map(|(l, r)| (l ^ r).count_ones())
        .sum::<u32>();

    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn hamming_q1_blocks_avx2(lhs_bits: &[u8], rhs_bits: &[u8]) -> u32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_xor_si256};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_xor_si256};

    let chunk_bytes = 32;
    let full_chunks = lhs_bits.len() / chunk_bytes;
    let mut sum = 0u32;

    for i in 0..full_chunks {
        let offset = i * chunk_bytes;
        let l = unsafe { _mm256_loadu_si256(lhs_bits.as_ptr().add(offset) as *const __m256i) };
        let r = unsafe { _mm256_loadu_si256(rhs_bits.as_ptr().add(offset) as *const __m256i) };
        let xored = _mm256_xor_si256(l, r);
        let mut lanes = [0u64; 4];
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, xored) };
        sum += lanes.iter().map(|v| v.count_ones()).sum::<u32>();
    }

    let tail_start = full_chunks * chunk_bytes;
    sum += lhs_bits[tail_start..]
        .iter()
        .zip(rhs_bits[tail_start..].iter())
        .map(|(l, r)| (l ^ r).count_ones())
        .sum::<u32>();

    sum
}

fn hamming_q1_blocks_impl(lhs_bits: &[u8], rhs_bits: &[u8]) -> PyResult<u32> {
    if lhs_bits.len() != rhs_bits.len() {
        return Err(PyValueError::new_err(format!(
            "shape mismatch: lhs={} rhs={}", lhs_bits.len(), rhs_bits.len()
        )));
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        return Ok(unsafe { hamming_q1_blocks_avx2(lhs_bits, rhs_bits) });
    }

    Ok(hamming_q1_blocks_scalar(lhs_bits, rhs_bits))
}

#[pyfunction]
fn hamming_q1_blocks(lhs_bits: &[u8], rhs_bits: &[u8]) -> PyResult<u32> {
    hamming_q1_blocks_impl(lhs_bits, rhs_bits)
}

// ---------------------------------------------------------------------------
// Top-k by Hamming similarity — rayon-parallel (from rs_quant_kernels)
// ---------------------------------------------------------------------------

#[pyfunction]
fn q1_batch_hamming_topk(
    query_bits: &[u8],
    packed_candidate_bits: &[u8],
    candidate_width: usize,
    k: usize,
) -> PyResult<Vec<(usize, u32)>> {
    if k == 0 || candidate_width == 0 || query_bits.len() != candidate_width {
        return Ok(Vec::new());
    }
    if packed_candidate_bits.is_empty() || packed_candidate_bits.len() % candidate_width != 0 {
        return Ok(Vec::new());
    }

    let candidate_count = packed_candidate_bits.len() / candidate_width;
    if k > candidate_count {
        return Err(PyValueError::new_err(format!(
            "k={} exceeds candidate_count={}", k, candidate_count
        )));
    }

    let total_bits = (query_bits.len() * 8) as u32;
    let mut scored: Vec<(usize, u32)> = packed_candidate_bits
        .par_chunks(candidate_width)
        .enumerate()
        .map(|(idx, cand)| {
            let distance = hamming_q1_blocks_impl(query_bits, cand)
                .unwrap_or(total_bits);
            (idx, total_bits.saturating_sub(distance))
        })
        .collect();

    let score_cmp = |a: &(usize, u32), b: &(usize, u32)| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0));
    if k < scored.len() {
        let split = k - 1;
        scored.select_nth_unstable_by(split, score_cmp);
        scored.truncate(k);
    }
    scored.sort_by(score_cmp);
    Ok(scored)
}

// ---------------------------------------------------------------------------
// Walsh-Hadamard Transform — 32-point in-place (from rs_quant_kernels)
// ---------------------------------------------------------------------------

#[pyfunction]
fn wht32(values: Vec<f32>) -> PyResult<Vec<f32>> {
    if values.len() != 32 {
        return Err(PyValueError::new_err(format!(
            "expected 32 values, got {}", values.len()
        )));
    }
    let mut arr: [f32; 32] = values.try_into().unwrap();
    let mut step = 1usize;
    while step < 32 {
        let jump = step * 2;
        let mut i = 0usize;
        while i < 32 {
            for j in 0..step {
                let a = arr[i + j];
                let b = arr[i + j + step];
                arr[i + j] = a + b;
                arr[i + j + step] = a - b;
            }
            i += jump;
        }
        step = jump;
    }
    let norm = 1.0f32 / 32.0f32.sqrt();
    for v in arr.iter_mut() {
        *v *= norm;
    }
    Ok(arr.to_vec())
}

// ---------------------------------------------------------------------------
// Cosine similarity — single and batch (from rs_cosine_similarity)
// ---------------------------------------------------------------------------

#[inline]
fn cosine_similarity_row(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut norm_a_sq = 0.0f64;
    let mut norm_b_sq = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a_sq += x * x;
        norm_b_sq += y * y;
    }
    if norm_a_sq == 0.0 || norm_b_sq == 0.0 {
        return 0.0;
    }
    dot / (norm_a_sq.sqrt() * norm_b_sq.sqrt())
}

#[pyfunction]
fn cosine_similarity(a: Vec<f64>, b: Vec<f64>) -> f64 {
    cosine_similarity_row(&a, &b)
}

#[pyfunction]
fn cosine_similarity_batch(left: Vec<Vec<f64>>, right: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    if left.is_empty() || right.is_empty() {
        return Ok(vec![vec![]; left.len()]);
    }
    let left_width = left[0].len();
    let right_width = right[0].len();
    if left.iter().any(|row| row.len() != left_width) {
        return Err(PyValueError::new_err("all left rows must have the same length"));
    }
    if right.iter().any(|row| row.len() != right_width) {
        return Err(PyValueError::new_err("all right rows must have the same length"));
    }
    if left_width != right_width {
        return Err(PyValueError::new_err("left and right batches must have the same feature width"));
    }
    let mut out = Vec::with_capacity(left.len());
    for left_row in &left {
        let mut row = Vec::with_capacity(right.len());
        for right_row in &right {
            row.push(cosine_similarity_row(left_row, right_row));
        }
        out.push(row);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// SimHash64 — Blake2b-based fingerprinting (from rs_hashing)
// ---------------------------------------------------------------------------

#[pyfunction]
fn simhash64(features: Vec<String>) -> String {
    let mut weights = [0isize; 64];
    for feature in features {
        let mut hasher = Blake2b64::new();
        hasher.update(feature.as_bytes());
        let digest = hasher.finalize();
        let mut value = 0u64;
        for i in 0..8 {
            value = (value << 8) | (digest[i] as u64);
        }
        for bit in 0..64 {
            if (value & (1 << bit)) != 0 {
                weights[bit] += 1;
            } else {
                weights[bit] -= 1;
            }
        }
    }
    let mut fingerprint = 0u64;
    for (bit, &weight) in weights.iter().enumerate() {
        if weight > 0 {
            fingerprint |= 1 << bit;
        }
    }
    format!("{:016x}", fingerprint)
}

#[pyfunction]
fn hamming_distance_u64_batch(lhs: Vec<u64>, rhs: Vec<u64>) -> PyResult<Vec<u32>> {
    if lhs.len() != rhs.len() {
        return Err(PyValueError::new_err(format!(
            "lhs and rhs must have equal length (got {} and {})", lhs.len(), rhs.len()
        )));
    }
    Ok(lhs.into_iter().zip(rhs.into_iter()).map(|(l, r)| (l ^ r).count_ones()).collect())
}

// ---------------------------------------------------------------------------
// Plutchik emotion model — 11-dimension VAD scoring (from rs_scoring)
// ---------------------------------------------------------------------------

#[pyfunction]
fn plutchik_valence_11d(v: Vec<f64>) -> PyResult<f64> {
    if v.len() != 11 {
        return Err(PyValueError::new_err(format!("expected 11 values, got {}", v.len())));
    }
    let pos = (v[0] + v[1] + v[7] + v[8]) / 4.0;
    let neg = (v[4] + v[5] + v[6] + v[9]) / 4.0;
    Ok((pos - neg).clamp(-1.0, 1.0))
}

#[pyfunction]
fn plutchik_arousal_11d(v: Vec<f64>) -> PyResult<f64> {
    if v.len() != 11 {
        return Err(PyValueError::new_err(format!("expected 11 values, got {}", v.len())));
    }
    Ok((v[2] + v[6] + v[3] + v[0] + v[7] + v[10]) / 6.0)
}

#[pyfunction]
fn plutchik_dominance_11d(v: Vec<f64>) -> PyResult<f64> {
    if v.len() != 11 {
        return Err(PyValueError::new_err(format!("expected 11 values, got {}", v.len())));
    }
    Ok((v[8] + v[7] + v[1]) / 3.0)
}

// ---------------------------------------------------------------------------
// Scoring utilities (from rs_scoring)
// ---------------------------------------------------------------------------

#[pyfunction]
fn smooth_vector(
    prior: Vec<f64>,
    observed: Vec<f64>,
    confidence: f64,
    alpha: f64,
    decay: f64,
) -> PyResult<Vec<f64>> {
    if prior.len() != observed.len() {
        return Err(PyValueError::new_err(format!(
            "prior and observed must have equal length (got {} and {})", prior.len(), observed.len()
        )));
    }
    if prior.is_empty() {
        return Err(PyValueError::new_err("prior/observed vectors must not be empty"));
    }
    let blend = alpha * confidence;
    Ok(prior.into_iter().zip(observed.into_iter())
        .map(|(p, o)| (p * decay * (1.0 - blend) + o * blend).clamp(0.0, 1.0))
        .collect())
}

#[pyfunction]
fn compute_momentum(prior: Vec<f64>, current: Vec<f64>) -> PyResult<f64> {
    if prior.len() != current.len() {
        return Err(PyValueError::new_err(format!(
            "prior and current must have equal length (got {} and {})", prior.len(), current.len()
        )));
    }
    if prior.is_empty() {
        return Err(PyValueError::new_err("prior/current vectors must not be empty"));
    }
    let mut sq_sum = 0.0;
    let mut prior_intensity = 0.0;
    let mut current_intensity = 0.0;
    for (p, c) in prior.into_iter().zip(current.into_iter()) {
        let d = c - p;
        sq_sum += d * d;
        prior_intensity += p;
        current_intensity += c;
    }
    let direction = if current_intensity >= prior_intensity { 1.0 } else { -1.0 };
    Ok((direction * sq_sum.sqrt() / 2.0).clamp(-1.0, 1.0))
}

#[pyfunction]
fn temporal_decay_score(
    timestamp: f64,
    now: f64,
    half_life_days: f64,
    method: &str,
) -> PyResult<f64> {
    if half_life_days <= 0.0 {
        return Err(PyValueError::new_err("half_life_days must be > 0"));
    }
    let age_seconds = now - timestamp;
    if age_seconds <= 0.0 {
        return Ok(1.0);
    }
    let age_days = age_seconds / 86_400.0;
    match method {
        "quadratic" => Ok(1.0 / (1.0 + (age_days / half_life_days).powi(2))),
        "exponential" => Ok((-age_days / half_life_days).exp()),
        _ => Err(PyValueError::new_err(format!(
            "Unknown decay method {method:?}; expected 'quadratic' or 'exponential'"
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (vector_scores, decay_scores, lexical_scores=None, alpha=0.65, beta=0.20, gamma=0.15))]
fn combined_retrieval_score_batch(
    vector_scores: Vec<f64>,
    decay_scores: Vec<f64>,
    lexical_scores: Option<Vec<f64>>,
    alpha: f64,
    beta: f64,
    gamma: f64,
) -> PyResult<Vec<f64>> {
    let weight_sum = alpha + beta + gamma;
    if (weight_sum - 1.0).abs() > 0.01 {
        return Err(PyValueError::new_err(format!(
            "Weights must sum to ~1.0 (got {weight_sum:.4})"
        )));
    }
    if vector_scores.len() != decay_scores.len() {
        return Err(PyValueError::new_err("vector_scores and decay_scores must have equal length"));
    }
    let lexical = lexical_scores.unwrap_or_else(|| vec![0.0; vector_scores.len()]);
    if lexical.len() != vector_scores.len() {
        return Err(PyValueError::new_err("lexical_scores must match vector_scores length"));
    }
    Ok(vector_scores.into_iter()
        .zip(decay_scores.into_iter())
        .zip(lexical.into_iter())
        .map(|((vs, ds), ls)| alpha * vs + beta * ds + gamma * ls)
        .collect())
}

#[pyfunction]
fn rrf_accumulate(
    result_keys: Vec<Vec<String>>,
    weights: Vec<f64>,
    k: usize,
) -> PyResult<HashMap<String, f64>> {
    if result_keys.is_empty() {
        return Ok(HashMap::new());
    }
    if result_keys.len() != weights.len() {
        return Err(PyValueError::new_err("weights must match result_keys length"));
    }
    let mut scores: HashMap<String, f64> = HashMap::new();
    for (set_idx, keys) in result_keys.iter().enumerate() {
        let w = weights[set_idx];
        for (rank, key) in keys.iter().enumerate() {
            let score = w / (k as f64 + rank as f64 + 1.0);
            *scores.entry(key.clone()).or_insert(0.0) += score;
        }
    }
    Ok(scores)
}

// ---------------------------------------------------------------------------
// PCM bytes ↔ audio (from rs_audio_core)
// ---------------------------------------------------------------------------

#[pyfunction]
fn pcm_bytes_to_audio(py: Python, pcm: &[u8]) -> Py<PyAny> {
    let mut audio = Vec::with_capacity(pcm.len() / 2);
    for chunk in pcm.chunks_exact(2) {
        let val = i16::from_le_bytes([chunk[0], chunk[1]]);
        audio.push(val as f32 / 32767.0);
    }
    Array1::from_vec(audio).into_pyarray(py).into_any().unbind()
}

// ---------------------------------------------------------------------------
// Levenshtein distance — edit distance with early termination (from rs_audio_core)
// ---------------------------------------------------------------------------

fn levenshtein_distance_impl(a: &str, b: &str, max_dist: isize) -> isize {
    if a == b {
        return 0;
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let (a_vec, b_vec) = if a_chars.len() < b_chars.len() {
        (b_chars, a_chars)
    } else {
        (a_chars, b_chars)
    };
    if b_vec.is_empty() {
        return a_vec.len() as isize;
    }
    let mut prefix_len = 0;
    while prefix_len < b_vec.len() && a_vec[prefix_len] == b_vec[prefix_len] {
        prefix_len += 1;
    }
    let a_sliced = &a_vec[prefix_len..];
    let b_sliced_full = &b_vec[prefix_len..];
    if b_sliced_full.is_empty() {
        return a_sliced.len() as isize;
    }
    let mut a_len = a_sliced.len();
    let mut b_len = b_sliced_full.len();
    while b_len > 0 && a_sliced[a_len - 1] == b_sliced_full[b_len - 1] {
        a_len -= 1;
        b_len -= 1;
    }
    let a_sliced = &a_sliced[..a_len];
    let b_sliced = &b_sliced_full[..b_len];
    if b_len == 0 {
        return a_len as isize;
    }
    if max_dist > -1 && (a_len as isize - b_len as isize).abs() > max_dist {
        return max_dist + 1;
    }
    let mut previous_row: Vec<isize> = (0..=b_len as isize).collect();
    let mut current_row: Vec<isize> = vec![0; b_len + 1];
    for (i, c1) in a_sliced.iter().enumerate() {
        current_row[0] = (i + 1) as isize;
        let mut min_current = current_row[0];
        for (j, c2) in b_sliced.iter().enumerate() {
            let val = if c1 == c2 {
                previous_row[j]
            } else {
                let min_val = previous_row[j]
                    .min(current_row[j])
                    .min(previous_row[j + 1]);
                min_val + 1
            };
            current_row[j + 1] = val;
            if val < min_current {
                min_current = val;
            }
        }
        if max_dist > -1 && min_current > max_dist {
            return max_dist + 1;
        }
        std::mem::swap(&mut previous_row, &mut current_row);
    }
    previous_row[b_len]
}

#[pyfunction]
#[pyo3(signature = (a, b, max_dist=-1))]
fn levenshtein_distance(a: &str, b: &str, max_dist: isize) -> isize {
    levenshtein_distance_impl(a, b, max_dist)
}

#[pyfunction]
fn levenshtein_ratio(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let dist = levenshtein_distance_impl(a, b, -1);
    let len_a = a.chars().count();
    let len_b = b.chars().count();
    let max_len = len_a.max(len_b);
    if max_len == 0 {
        return 1.0;
    }
    (max_len as f64 - dist as f64) / max_len as f64
}

// ---------------------------------------------------------------------------
// Temporal decay — batch variants (from rs_scoring)
// ---------------------------------------------------------------------------

#[pyfunction]
fn temporal_decay_batch(
    timestamps: Vec<f64>,
    now: f64,
    half_life_days: f64,
    method: &str,
) -> PyResult<Vec<f64>> {
    if half_life_days <= 0.0 {
        return Err(PyValueError::new_err("half_life_days must be > 0"));
    }
    let mut out = Vec::with_capacity(timestamps.len());
    for ts in timestamps {
        let age_seconds = now - ts;
        let score = if age_seconds <= 0.0 {
            1.0
        } else {
            let age_days = age_seconds / 86_400.0;
            match method {
                "quadratic" => 1.0 / (1.0 + (age_days / half_life_days).powi(2)),
                "exponential" => (-age_days / half_life_days).exp(),
                _ => return Err(PyValueError::new_err(format!(
                    "Unknown decay method {method:?}; expected 'quadratic' or 'exponential'"
                ))),
            }
        };
        out.push(score);
    }
    Ok(out)
}

#[pyfunction]
fn batch_decay_scores(
    items: Vec<HashMap<String, f64>>,
    timestamp_key: &str,
    now: f64,
    half_life_days: f64,
    method: &str,
) -> PyResult<Vec<f64>> {
    let mut timestamps = Vec::with_capacity(items.len());
    for item in &items {
        let ts = item.get(timestamp_key).ok_or_else(|| {
            PyValueError::new_err(format!("missing timestamp key {timestamp_key:?}"))
        })?;
        timestamps.push(*ts);
    }
    temporal_decay_batch(timestamps, now, half_life_days, method)
}

// Module registration

#[pymodule]
fn rs_codec(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Audio codec
    m.add_function(wrap_pyfunction!(decode_raw_f32, m)?)?;
    m.add_function(wrap_pyfunction!(decode_b64_f32, m)?)?;
    m.add_function(wrap_pyfunction!(encode_wav_b64, m)?)?;
    m.add_function(wrap_pyfunction!(float_to_int16_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(crossfade_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(pcm_bytes_to_audio, m)?)?;
    // DSP kernels
    m.add_function(wrap_pyfunction!(soft_compressor, m)?)?;
    m.add_function(wrap_pyfunction!(agc_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(iir_1pole_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(highpass_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(fused_preprocess_for_vad_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_rms, m)?)?;
    // Sentence splitting & text matching
    m.add_class::<SentenceSplitter>()?;
    m.add_function(wrap_pyfunction!(levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_ratio, m)?)?;
    // Quantization & similarity
    m.add_function(wrap_pyfunction!(hamming_q1_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(q1_batch_hamming_topk, m)?)?;
    m.add_function(wrap_pyfunction!(wht32, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity_batch, m)?)?;
    // Hashing & fingerprinting
    m.add_function(wrap_pyfunction!(simhash64, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_distance_u64_batch, m)?)?;
    // Emotion / affect scoring
    m.add_function(wrap_pyfunction!(plutchik_valence_11d, m)?)?;
    m.add_function(wrap_pyfunction!(plutchik_arousal_11d, m)?)?;
    m.add_function(wrap_pyfunction!(plutchik_dominance_11d, m)?)?;
    // Retrieval scoring
    m.add_function(wrap_pyfunction!(smooth_vector, m)?)?;
    m.add_function(wrap_pyfunction!(compute_momentum, m)?)?;
    m.add_function(wrap_pyfunction!(temporal_decay_score, m)?)?;
    m.add_function(wrap_pyfunction!(temporal_decay_batch, m)?)?;
    m.add_function(wrap_pyfunction!(batch_decay_scores, m)?)?;
    m.add_function(wrap_pyfunction!(combined_retrieval_score_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rrf_accumulate, m)?)?;
    Ok(())
}
