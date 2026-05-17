# Changelog

All notable crate-specific changes for `rs_kv_quant_contracts` are recorded here.

## [Unreleased]

### Added

- Wave 31 (SMG positional-index contract): added `PositionalIndexKey`, `PositionalIndexEntry`, `PositionalIndexError`, and `PositionalIndexResult<T>` types for prefix-match routing bookkeeping derived from the SMG positional indexer pattern.
- Added `positional_index_contract_round_trips` unit test covering serde round-trip and typed error display strings.
- Added crate-local documentation coverage with `README.md` and `CHANGELOG.md` tracking for `rs_kv_quant_contracts` (0.1.0).
