use pyo3::prelude::*;
use pyo3::types::PyList;

#[pyclass]
pub struct ReasoningFilter {
    state: u8,
    buf: String,
}

#[pymethods]
impl ReasoningFilter {
    #[new]
    fn new() -> Self {
        ReasoningFilter {
            state: 0,
            buf: String::new(),
        }
    }

    fn process<'py>(&mut self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyList>> {
        let results = PyList::empty_bound(py);

        if self.state == 0 {
            self.buf.push_str(text);
            if let Some(think_idx) = self.buf.find("<think>") {
                let before = &self.buf[..think_idx];
                if !before.is_empty() {
                    results.append(("content", before))?;
                }
                self.state = 1;

                let rest = self.buf[think_idx + "<think>".len()..].to_string();
                self.buf = rest;

                if let Some(end_idx) = self.buf.find("</think>") {
                    let reasoning = &self.buf[..end_idx];
                    let mut after = &self.buf[end_idx + "</think>".len()..];
                    if after.starts_with('\n') {
                        after = &after[1..];
                    }

                    if !reasoning.is_empty() {
                        results.append(("reasoning_content", reasoning))?;
                    }
                    self.state = 2;

                    if !after.is_empty() {
                        results.append(("content", after))?;
                    }
                    self.buf.clear();
                } else if !self.buf.is_empty() {
                    results.append(("reasoning_content", &self.buf))?;
                    self.buf.clear();
                }
            } else if self.buf.len() > 7 && !self.buf.contains('<') {
                results.append(("content", &self.buf))?;
                self.buf.clear();
            }
        } else if self.state == 1 {
            self.buf.push_str(text);
            if let Some(end_idx) = self.buf.find("</think>") {
                let reasoning = &self.buf[..end_idx];
                let mut after = &self.buf[end_idx + "</think>".len()..];
                if after.starts_with('\n') {
                    after = &after[1..];
                }

                if !reasoning.is_empty() {
                    results.append(("reasoning_content", reasoning))?;
                }
                self.state = 2;

                if !after.is_empty() {
                    results.append(("content", after))?;
                }
                self.buf.clear();
            } else {
                results.append(("reasoning_content", &self.buf))?;
                self.buf.clear();
            }
        } else {
            if !text.is_empty() {
                results.append(("content", text))?;
            }
        }

        Ok(results)
    }

    fn flush<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let results = PyList::empty_bound(py);
        if !self.buf.is_empty() {
            if self.state == 0 {
                results.append(("content", &self.buf))?;
            } else if self.state == 1 {
                results.append(("reasoning_content", &self.buf))?;
            }
            self.buf.clear();
        }
        Ok(results)
    }
}
