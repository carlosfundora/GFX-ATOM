mod reasoning;

use ignore::WalkBuilder;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use xxhash_rust::xxh64::Xxh64;
use xxhash_rust::xxh3::xxh3_128;

#[pyfunction]
fn compute_hash(token_ids: Vec<i64>, prefix: i128) -> u64 {
    let mut hasher = Xxh64::new(0);
    if prefix != -1 {
        // since prefix comes from python hash, it might be large.
        hasher.update(&(prefix as u64).to_le_bytes());
    }
    // numpy array tobytes() for int64 is just the raw bytes
    // Since token_ids in python is often list[int], let's treat them as i64.
    for &token in &token_ids {
        hasher.update(&token.to_le_bytes());
    }
    hasher.digest()
}

#[pyfunction]
fn compute_string_hash(content: &str) -> String {
    let hash = xxh3_128(content.as_bytes());
    format!("{:032x}", hash)
}

#[pyfunction]
fn compute_bytes_hash(content: &Bound<'_, PyBytes>) -> String {
    let hash = xxh3_128(content.as_bytes());
    format!("{:032x}", hash)
}

#[pyfunction]
fn find_files(path: &str) -> PyResult<Vec<String>> {
    let mut files = Vec::new();
    let walker = WalkBuilder::new(path)
        .follow_links(true)
        .hidden(false)
        .git_ignore(false)
        .git_global(false)
        .git_exclude(false)
        .parents(false)
        .build();

    for result in walker {
        match result {
            Ok(entry) => {
                if entry
                    .file_type()
                    .map(|file_type| file_type.is_file())
                    .unwrap_or(false)
                {
                    if let Some(path_str) = entry.path().to_str() {
                        files.push(path_str.to_string());
                    }
                }
            }
            Err(err) => return Err(pyo3::exceptions::PyIOError::new_err(err.to_string())),
        }
    }

    Ok(files)
}

#[pyfunction]
fn parse_tool_calls(text: &str) -> PyResult<(String, Vec<PyObject>)> {
    Python::with_gil(|py| {
        let mut tool_calls = Vec::new();
        let mut content = text.to_string();

        if let Some(start_idx) = text.find("<|tool_calls_section_begin|>") {
            let before = &text[..start_idx];
            let rest = &text[start_idx + "<|tool_calls_section_begin|>".len()..];

            if let Some(end_idx) = rest.find("<|tool_calls_section_end|>") {
                let after = &rest[end_idx + "<|tool_calls_section_end|>".len()..];
                content = format!("{}{}", before, after);

                let section_text = &rest[..end_idx];
                tool_calls = parse_tool_call_entries(py, section_text)?;
            } else {
                content = before.to_string();
                tool_calls = parse_tool_call_entries(py, rest)?;
            }
        }

        Ok((content.trim().to_string(), tool_calls))
    })
}

fn parse_tool_call_entries(py: Python, section_text: &str) -> PyResult<Vec<PyObject>> {
    let mut tool_calls = Vec::new();
    let mut current_idx = 0;

    while let Some(begin_idx) = section_text[current_idx..].find("<|tool_call_begin|>") {
        let actual_begin = current_idx + begin_idx;
        let rest = &section_text[actual_begin + "<|tool_call_begin|>".len()..];

        if let Some(arg_begin_idx) = rest.find("<|tool_call_argument_begin|>") {
            let func_part = &rest[..arg_begin_idx];

            // Extract functions.NAME:INDEX
            if func_part.starts_with("functions.") {
                if let Some(colon_idx) = func_part.find(':') {
                    let name = &func_part["functions.".len()..colon_idx];

                    let arg_rest = &rest[arg_begin_idx + "<|tool_call_argument_begin|>".len()..];
                    if let Some(end_idx) = arg_rest.find("<|tool_call_end|>") {
                        let arguments = arg_rest[..end_idx].trim();

                        let uuid = uuid::Uuid::new_v4().simple().to_string();
                        let call_id = format!("call_{}", &uuid[..8]);

                        let tool_call = pyo3::types::PyDict::new_bound(py);
                        tool_call.set_item("id", call_id)?;
                        tool_call.set_item("type", "function")?;

                        let function_dict = pyo3::types::PyDict::new_bound(py);
                        function_dict.set_item("name", name)?;
                        function_dict.set_item("arguments", arguments)?;
                        tool_call.set_item("function", function_dict)?;

                        tool_calls.push(tool_call.into());

                        current_idx = actual_begin + "<|tool_call_begin|>".len() + arg_begin_idx + "<|tool_call_argument_begin|>".len() + end_idx + "<|tool_call_end|>".len();
                        continue;
                    }
                }
            }
        }
        // If we didn't match the whole pattern, just advance past the begin token to avoid infinite loop
        current_idx = actual_begin + "<|tool_call_begin|>".len();
    }

    Ok(tool_calls)
}


#[pymodule]
fn atom_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_hash, m)?)?;
    m.add_function(wrap_pyfunction!(compute_string_hash, m)?)?;
    m.add_function(wrap_pyfunction!(compute_bytes_hash, m)?)?;
    m.add_function(wrap_pyfunction!(find_files, m)?)?;
    m.add_function(wrap_pyfunction!(parse_tool_calls, m)?)?;
    m.add_class::<reasoning::ReasoningFilter>()?;
    Ok(())
}
