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

#[pymodule]
fn atom_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_hash, m)?)?;
    Ok(())
}
