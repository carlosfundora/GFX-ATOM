# Changelog

All notable crate-specific changes for `rs_kv_quant_contracts` are recorded here.

## [Unreleased]

### Added

- Wave 32 (FP8 KV alignment contract): added `validate_fp8_kv_dimension()` and `align_dimension_to_16()` helper functions to enforce 16-byte alignment constraints for FP8 KV caches, aligning with upstream ATOM commit 10fba75.
- Added `KvCodecError::Fp8DimensionMisaligned` error variant for explicit FP8 dimension validation failures.
- Added `fp8_kv_dimension_alignment_validation()` and `align_dimension_to_16_helper()` unit tests covering alignment computation and error handling.
- Derived `Clone` on `KvCodecError` and `PositionalIndexError` for better error ergonomics.

### Fixed

- Fixed error handling in FP8 validation to support idiomatic Result-based control flow.

## [Previous]

### Added

- Wave 31 (SMG positional-index contract): added `PositionalIndexKey`, `PositionalIndexEntry`, `PositionalIndexError`, and `PositionalIndexResult<T>` types for prefix-match routing bookkeeping derived from the SMG positional indexer pattern.
- Added `positional_index_contract_round_trips` unit test covering serde round-trip and typed error display strings.
- Added crate-local documentation coverage with `README.md` and `CHANGELOG.md` tracking for `rs_kv_quant_contracts` (0.1.0).
