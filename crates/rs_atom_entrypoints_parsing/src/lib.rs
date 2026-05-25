use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict, PyTuple, PyString};
use regex::Regex;
use once_cell::sync::Lazy;
use uuid::Uuid;

static SECTION_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>").unwrap()
});
static UNCLOSED_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)<\|tool_calls_section_begin\|>(.*?)$").unwrap()
});
static ENTRY_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)<\|tool_call_begin\|>functions\.(\w+):(\d+)<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>").unwrap()
});

#[pyclass]
pub struct ToolCallStreamParser {
    state: i32,
    buf: String,
    emitted_calls: i32,
}

#[pymethods]
impl ToolCallStreamParser {
    #[new]
    fn new() -> Self {
        Self {
            state: 0,
            buf: String::new(),
            emitted_calls: 0,
        }
    }

    fn process<'py>(&mut self, py: Python<'py>, text: &str) -> PyResult<&'py PyList> {
        let results = PyList::empty(py);

        if self.state == 0 {
            self.buf.push_str(text);
            if let Some(idx) = self.buf.find("<|tool_calls_section_begin|>") {
                let before = self.buf[..idx].to_string();
                if !before.is_empty() {
                    let tuple = ("content", before);
                    results.append(tuple)?;
                }
                self.state = 1;
                self.buf = self.buf[idx + "<|tool_calls_section_begin|>".len()..].to_string();
                self._process_buffer(py, results)?;
            } else if !self.buf.contains("<|tool") && self.buf.len() > 30 {
                let tuple = ("content", self.buf.clone());
                results.append(tuple)?;
                self.buf.clear();
            }
        } else if self.state == 1 {
            self.buf.push_str(text);
            if let Some(idx) = self.buf.find("<|tool_calls_section_end|>") {
                let remaining = self.buf[..idx].to_string();
                let after = self.buf[idx + "<|tool_calls_section_end|>".len()..].to_string();
                self.buf = remaining;
                self._process_buffer(py, results)?;
                let tuple = ("tool_call_end", py.None());
                results.append(tuple)?;
                self.state = 2;
                self.buf = after;
            } else {
                self._process_buffer(py, results)?;
            }
        }

        Ok(results)
    }

    fn flush<'py>(&mut self, py: Python<'py>) -> PyResult<&'py PyList> {
        let results = PyList::empty(py);
        if self.state == 0 && !self.buf.is_empty() {
            let tuple = ("content", self.buf.clone());
            results.append(tuple)?;
            self.buf.clear();
        } else if self.state == 1 {
            self._process_buffer(py, results)?;
            if self.emitted_calls > 0 {
                let tuple = ("tool_call_end", py.None());
                results.append(tuple)?;
            }
        }
        Ok(results)
    }
}

impl ToolCallStreamParser {
    fn _process_buffer<'py>(&mut self, py: Python<'py>, results: &'py PyList) -> PyResult<()> {
        while self.buf.contains("<|tool_call_begin|>") && self.buf.contains("<|tool_call_end|>") {
            if let Some(caps) = ENTRY_REGEX.captures(&self.buf) {
                let m = caps.get(0).unwrap();
                let name = caps.get(1).unwrap().as_str();
                let index: i32 = caps.get(2).unwrap().as_str().parse().unwrap_or(0);
                let arguments = caps.get(3).unwrap().as_str().trim();

                let call_id = format!("call_{}", &Uuid::new_v4().to_string().replace("-", "")[..8]);

                let func_dict = PyDict::new(py);
                func_dict.set_item("name", name)?;
                func_dict.set_item("arguments", "")?;

                let start_dict = PyDict::new(py);
                start_dict.set_item("index", index)?;
                start_dict.set_item("id", call_id)?;
                start_dict.set_item("type", "function")?;
                start_dict.set_item("function", func_dict)?;

                let start_tuple = ("tool_call_start", start_dict);
                results.append(start_tuple)?;

                if !arguments.is_empty() {
                    let arg_func_dict = PyDict::new(py);
                    arg_func_dict.set_item("arguments", arguments)?;

                    let args_dict = PyDict::new(py);
                    args_dict.set_item("index", index)?;
                    args_dict.set_item("function", arg_func_dict)?;

                    let args_tuple = ("tool_call_args", args_dict);
                    results.append(args_tuple)?;
                }

                self.buf = self.buf[m.end()..].to_string();
                self.emitted_calls += 1;
            } else {
                break;
            }
        }
        Ok(())
    }
}

#[pyfunction]
fn parse_tool_calls<'py>(py: Python<'py>, text: &str) -> PyResult<&'py PyTuple> {
    if let Some(caps) = SECTION_REGEX.captures(text) {
        let m = caps.get(0).unwrap();
        let content = &text[..m.start()];
        let section_text = caps.get(1).unwrap().as_str();
        let tool_calls = _parse_tool_call_entries(py, section_text)?;
        let content_py: Py<PyString> = PyString::new(py, content.trim()).into_py(py);
        let tool_calls_py: Py<PyList> = tool_calls.into_py(py);
        return Ok(PyTuple::new(py, &[content_py.into_py(py), tool_calls_py.into_py(py)]));
    } else if let Some(caps) = UNCLOSED_REGEX.captures(text) {
        let m = caps.get(0).unwrap();
        let content = &text[..m.start()];
        let section_text = caps.get(1).unwrap().as_str();
        let tool_calls = _parse_tool_call_entries(py, section_text)?;
        let content_py: Py<PyString> = PyString::new(py, content.trim()).into_py(py);
        let tool_calls_py: Py<PyList> = tool_calls.into_py(py);
        return Ok(PyTuple::new(py, &[content_py.into_py(py), tool_calls_py.into_py(py)]));
    }

    let content_py: Py<PyString> = PyString::new(py, text).into_py(py);
    let tool_calls_py: Py<PyList> = PyList::empty(py).into_py(py);
    Ok(PyTuple::new(py, &[content_py.into_py(py), tool_calls_py.into_py(py)]))
}

fn _parse_tool_call_entries<'py>(py: Python<'py>, section_text: &str) -> PyResult<&'py PyList> {
    let list = PyList::empty(py);
    for caps in ENTRY_REGEX.captures_iter(section_text) {
        let name = caps.get(1).unwrap().as_str();
        let arguments = caps.get(3).unwrap().as_str().trim();
        let call_id = format!("call_{}", &Uuid::new_v4().to_string().replace("-", "")[..8]);

        let func_dict = PyDict::new(py);
        func_dict.set_item("name", name)?;
        func_dict.set_item("arguments", arguments)?;

        let tool_call = PyDict::new(py);
        tool_call.set_item("id", call_id)?;
        tool_call.set_item("type", "function")?;
        tool_call.set_item("function", func_dict)?;

        list.append(tool_call)?;
    }
    Ok(list)
}

#[pyclass]
pub struct ReasoningFilter {
    state: i32,
    buf: String,
}

#[pymethods]
impl ReasoningFilter {
    #[new]
    fn new() -> Self {
        Self {
            state: 0,
            buf: String::new(),
        }
    }

    fn process<'py>(&mut self, py: Python<'py>, text: &str) -> PyResult<&'py PyList> {
        let results = PyList::empty(py);

        if self.state == 0 {
            self.buf.push_str(text);
            if let Some(idx) = self.buf.find("<think>") {
                let before = self.buf[..idx].to_string();
                if !before.is_empty() {
                    let tuple = ("content", before);
                    results.append(tuple)?;
                }
                self.state = 1;
                self.buf = self.buf[idx + "<think>".len()..].to_string();

                if let Some(end_idx) = self.buf.find("</think>") {
                    let reasoning = self.buf[..end_idx].to_string();
                    let after = self.buf[end_idx + "</think>".len()..].trim_start_matches('\n').to_string();
                    if !reasoning.is_empty() {
                        let tuple = ("reasoning_content", reasoning);
                        results.append(tuple)?;
                    }
                    self.state = 2;
                    self.buf.clear();
                    if !after.is_empty() {
                        let tuple = ("content", after);
                        results.append(tuple)?;
                    }
                } else if !self.buf.is_empty() {
                    let tuple = ("reasoning_content", self.buf.clone());
                    results.append(tuple)?;
                    self.buf.clear();
                }
            } else if let Some(idx) = self.buf.find("</think>") {
                let reasoning = self.buf[..idx].to_string();
                let after = self.buf[idx + "</think>".len()..].trim_start_matches('\n').to_string();
                if !reasoning.is_empty() {
                    let tuple = ("reasoning_content", reasoning);
                    results.append(tuple)?;
                }
                self.state = 2;
                self.buf.clear();
                if !after.is_empty() {
                    let tuple = ("content", after);
                    results.append(tuple)?;
                }
            } else if !self.buf.contains("<") && self.buf.len() > 7 {
                let tuple = ("content", self.buf.clone());
                results.append(tuple)?;
                self.buf.clear();
            }
        } else if self.state == 1 {
            self.buf.push_str(text);
            if let Some(idx) = self.buf.find("</think>") {
                let reasoning = self.buf[..idx].to_string();
                let after = self.buf[idx + "</think>".len()..].trim_start_matches('\n').to_string();
                if !reasoning.is_empty() {
                    let tuple = ("reasoning_content", reasoning);
                    results.append(tuple)?;
                }
                self.state = 2;
                self.buf.clear();
                if !after.is_empty() {
                    let tuple = ("content", after);
                    results.append(tuple)?;
                }
            } else {
                let tuple = ("reasoning_content", self.buf.clone());
                results.append(tuple)?;
                self.buf.clear();
            }
        } else {
            if !text.is_empty() {
                let tuple = ("content", text.to_string());
                results.append(tuple)?;
            }
        }

        Ok(results)
    }

    fn flush<'py>(&mut self, py: Python<'py>) -> PyResult<&'py PyList> {
        let results = PyList::empty(py);
        if !self.buf.is_empty() {
            if self.state == 0 {
                let tuple = ("content", self.buf.clone());
                results.append(tuple)?;
            } else if self.state == 1 {
                let tuple = ("reasoning_content", self.buf.clone());
                results.append(tuple)?;
            }
            self.buf.clear();
        }
        Ok(results)
    }
}

#[pymodule]
fn atom_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ToolCallStreamParser>()?;
    m.add_class::<ReasoningFilter>()?;
    m.add_function(wrap_pyfunction!(parse_tool_calls, m)?)?;
    Ok(())
}
