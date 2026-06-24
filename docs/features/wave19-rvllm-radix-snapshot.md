# Wave-19 rvLLM radix snapshot

rvLLM’s compact radix-tree snapshot format was assimilated into the shared KV contract layer.

## Captured contract surface

- radix-tree snapshots with pre-order nodes
- tenant + access-epoch metadata per node
- compact bincode serialization
- node-count and serialized-edge-size helpers

## Integration result

- `rs_kv_quant_contracts` now exposes a portable radix snapshot type.
- `rs_kv_validation_harness` proves round-trip serialization on a representative snapshot.
- The contract is backend-neutral and can support prefix-cache export or replay tooling later.
