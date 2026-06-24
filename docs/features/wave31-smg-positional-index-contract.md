# Wave 31 — SMG Positional-Index Contract

## Donor
`smg` — positional indexer pattern from `event_tree.rs` / `kv_index/src/lib.rs`

## Problem
The shared KV contract crate already had content-hash and sequence-hash helpers, prefix-match results, and radix snapshot types. The missing routing primitive was a composite key that binds a block's position to its content hash, plus a typed record for which worker owns that entry and what sequence-hash it carries.

Without typed positional index contracts, any routing-level code that needs to distinguish "same content, different position" from "different content, same position" is forced to inline ad hoc structs or use untyped maps.

## What was added

### `rs_kv_quant_contracts`

| Type | Purpose |
|---|---|
| `PositionalIndexKey` | Composite `(position: usize, content_hash: ContentHash)` — position-aware block identity |
| `PositionalIndexEntry` | Full record: `key`, `sequence_hash`, `worker_id` |
| `PositionalIndexError` | Typed errors: `WorkerNotTracked(u32)`, `ParentBlockNotFound(usize, u32)` |
| `PositionalIndexResult<T>` | Type alias for `Result<T, PositionalIndexError>` |

All types derive `Debug`, `Clone`, `PartialEq`, `Eq`, and `Serialize` / `Deserialize`.

### `rs_kv_validation_harness`

Two new validation cases in `run_validation_suite()`:

- `positional_index_entry_round_trip` — serde JSON round-trip check
- `positional_index_error_shape` — `Display` string content checks for both error variants

## Design notes

- `ContentHash` is position-independent (xxh3 over token bytes).
- `SequenceHash` is position-aware (assigned by the serving backend, not computed locally).
- `PositionalIndexKey` combines both axes so cache hit/miss logic can test position equality independently from content equality.
- `PositionalIndexError` follows the existing `GraphError` / `KvCodecError` `thiserror`-derived pattern.

## Integration note

These types are routing primitives only. They do not modify KV allocation, eviction behavior, or backend dispatch. They are designed to be composed into higher-level routing layers (e.g., SGLang radix cache hit check, per-worker prefix validation) without requiring any runtime dependency beyond `serde` and `thiserror`.

## Tests

```
cargo test -p rs_kv_quant_contracts   # 8 passed
cargo test -p rs_kv_validation_harness # 2 passed
```
