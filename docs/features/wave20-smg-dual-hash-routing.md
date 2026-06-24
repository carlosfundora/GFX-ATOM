# Wave-20 SMG dual-hash routing

SMG’s dual-hash cache-aware routing helper was assimilated into the shared KV contract layer.

## Captured contract surface

- content hashes for token blocks
- request content-hash chunking
- deterministic page-aligned hashing
- zero-length block handling

## Integration result

- `rs_kv_quant_contracts` now exposes reusable content-hash and request-hash helpers.
- `rs_kv_validation_harness` proves chunking behavior across full and empty block paths.
- The contract stays backend-neutral and can support prefix reuse, routing, or cache index tooling.
