use ignore::WalkBuilder;
use pyo3::prelude::*;
use xxhash_rust::xxh64::Xxh64;

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

#[pymodule]
fn atom_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_hash, m)?)?;
    m.add_function(wrap_pyfunction!(find_files, m)?)?;
    Ok(())
}
