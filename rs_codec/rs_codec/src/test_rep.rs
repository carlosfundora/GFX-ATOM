use numpy::{PyArray2, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
fn np_rep_penalty(py: Python, scores: &Bound<'_, PyArray2<f32>>, input_ids: &Bound<'_, PyArray2<i64>>, penalty: f32) {
    let mut scores_mut = scores.as_array_mut();
    let ids_view = input_ids.as_array();

    let batch_size = ids_view.shape()[0];
    let seq_len = ids_view.shape()[1];

    for b in 0..batch_size {
        for i in 0..seq_len {
            let id = ids_view[[b, i]] as usize;
            let val = scores_mut[[b, id]];
            if val < 0.0 {
                scores_mut[[b, id]] = val * penalty;
            } else {
                scores_mut[[b, id]] = val / penalty;
            }
        }
    }
}
