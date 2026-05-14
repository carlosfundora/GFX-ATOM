use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use numpy::ndarray::Array1;

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
fn audio_to_pcm_bytes(py: Python<'_>, audio: PyReadonlyArray1<f32>) -> PyObject {
    let view = audio.as_array();
    let mut pcm_data = Vec::with_capacity(view.len() * 2);
    for &x in view.iter() {
        let val = (x * 32767.0).clamp(-32768.0, 32767.0) as i16;
        pcm_data.extend_from_slice(&val.to_le_bytes());
    }
    pyo3::types::PyBytes::new(py, &pcm_data).into()
}

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
                } else if c == '。' || c == '！' || c == '？' {
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

#[pymodule]
fn rs_codec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(soft_compressor, m)?)?;
    m.add_function(wrap_pyfunction!(agc_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(iir_1pole_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(highpass_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(audio_to_pcm_bytes, m)?)?;
    m.add_class::<SentenceSplitter>()?;
    Ok(())
}
