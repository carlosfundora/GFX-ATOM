use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyclass]
pub struct ToolCallStreamParser {
    state: u8,
    buf: String,
    emitted_calls: usize,
}

#[pymethods]
impl ToolCallStreamParser {
    #[new]
    fn new() -> Self {
        ToolCallStreamParser {
            state: 0,
            buf: String::new(),
            emitted_calls: 0,
        }
    }

    fn process<'py>(&mut self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyList>> {
        let results = PyList::empty_bound(py);

        if self.state == 0 {
            self.buf.push_str(text);
            if let Some(begin_idx) = self.buf.find("<|tool_calls_section_begin|>") {
                let before = &self.buf[..begin_idx];
                if !before.is_empty() {
                    results.append(("content", before))?;
                }
                self.state = 1;
                let rest = self.buf[begin_idx + "<|tool_calls_section_begin|>".len()..].to_string();
                self.buf = rest;

                let buf_results = self.process_buffer(py)?;
                for item in buf_results {
                    results.append(item)?;
                }
            } else if !self.buf.contains("<|tool") && self.buf.len() > 30 {
                results.append(("content", self.buf.clone()))?;
                self.buf.clear();
            }
        } else if self.state == 1 {
            self.buf.push_str(text);
            if let Some(end_idx) = self.buf.find("<|tool_calls_section_end|>") {
                let remaining = self.buf[..end_idx].to_string();
                self.buf = remaining;

                let buf_results = self.process_buffer(py)?;
                for item in buf_results {
                    results.append(item)?;
                }

                results.append(("tool_call_end", py.None()))?;
                self.state = 2;
                self.buf.clear();
            } else {
                let buf_results = self.process_buffer(py)?;
                for item in buf_results {
                    results.append(item)?;
                }
            }
        }

        Ok(results)
    }

    fn flush<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let results = PyList::empty_bound(py);
        if self.state == 0 && !self.buf.is_empty() {
            results.append(("content", self.buf.clone()))?;
            self.buf.clear();
        } else if self.state == 1 {
            let buf_results = self.process_buffer(py)?;
            for item in buf_results {
                results.append(item)?;
            }
            if self.emitted_calls > 0 {
                results.append(("tool_call_end", py.None()))?;
            }
        }
        Ok(results)
    }
}

impl ToolCallStreamParser {
    fn process_buffer<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<PyObject>> {
        let mut results = Vec::new();
        let mut current_idx = 0;

        while let Some(begin_idx) = self.buf[current_idx..].find("<|tool_call_begin|>") {
            let actual_begin = current_idx + begin_idx;
            let rest = &self.buf[actual_begin + "<|tool_call_begin|>".len()..];

            if let Some(arg_begin_idx) = rest.find("<|tool_call_argument_begin|>") {
                let func_part = &rest[..arg_begin_idx];

                if func_part.starts_with("functions.") {
                    if let Some(colon_idx) = func_part.find(':') {
                        let name = &func_part["functions.".len()..colon_idx];
                        if let Ok(index) = func_part[colon_idx + 1..].parse::<usize>() {
                            let arg_rest = &rest[arg_begin_idx + "<|tool_call_argument_begin|>".len()..];

                            if let Some(end_idx) = arg_rest.find("<|tool_call_end|>") {
                                let arguments = arg_rest[..end_idx].trim();

                                let uuid_str = uuid::Uuid::new_v4().simple().to_string();
                                let call_id = format!("call_{}", &uuid_str[..8]);

                                let start_data = PyDict::new_bound(py);
                                start_data.set_item("index", index)?;
                                start_data.set_item("id", call_id)?;
                                start_data.set_item("type", "function")?;

                                let func_dict = PyDict::new_bound(py);
                                func_dict.set_item("name", name)?;
                                func_dict.set_item("arguments", "")?;
                                start_data.set_item("function", func_dict)?;

                                results.push(("tool_call_start", start_data).into_py(py));

                                if !arguments.is_empty() {
                                    let arg_data = PyDict::new_bound(py);
                                    arg_data.set_item("index", index)?;

                                    let arg_func_dict = PyDict::new_bound(py);
                                    arg_func_dict.set_item("arguments", arguments)?;
                                    arg_data.set_item("function", arg_func_dict)?;

                                    results.push(("tool_call_args", arg_data).into_py(py));
                                }

                                let total_len = actual_begin
                                    + "<|tool_call_begin|>".len()
                                    + arg_begin_idx
                                    + "<|tool_call_argument_begin|>".len()
                                    + end_idx
                                    + "<|tool_call_end|>".len();

                                self.buf = self.buf[total_len..].to_string();
                                current_idx = 0;
                                self.emitted_calls += 1;
                                continue;
                            }
                        }
                    }
                }
            }

            // Advance past current token so we don't get stuck if it's malformed
            current_idx = actual_begin + "<|tool_call_begin|>".len();
        }

        Ok(results)
    }
}
