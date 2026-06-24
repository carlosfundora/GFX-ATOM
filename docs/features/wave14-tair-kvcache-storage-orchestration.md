# Wave-14 Tair KVCache storage orchestration

Tair KVCache was assimilated as a storage orchestration capability profile rather than a new runtime/backend wrapper.

## Captured capability surface

- distributed memory pooling
- dynamic multi-level caching
- global metadata management
- capacity management
- prefix matching
- sliding-window matching
- KV matching
- two-phase writes
- async eviction
- trace-replay optimization

## Integration result

- Engine runtime profiles in Python and Rust now expose storage orchestration capability flags.
- The capability surface stays read-only and fail-closed; it does not add a new scheduler or storage backend implementation.
- The new profile is covered by Python/Rust parity tests.
