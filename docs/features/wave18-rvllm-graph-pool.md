# Wave-18 rvLLM graph pool

rvLLM’s captured-graph pool and metadata-layout gate were assimilated into the shared KV contract layer.

## Captured contract surface

- metadata layouts keyed by bucket and max-block count
- layout hashing for replay-time drift detection
- captured graphs with fingerprint + layout-hash metadata
- typed graph-pool lookup and replay checks

## Integration result

- `rs_kv_quant_contracts` now exposes a portable graph-pool contract.
- `rs_kv_validation_harness` enforces matching-layout and drift-rejection behavior.
- The pattern stays backend-neutral: it captures the replay contract without embedding CUDA-specific capture code.
