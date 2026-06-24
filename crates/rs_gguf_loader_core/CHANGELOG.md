# Changelog

All notable crate-specific changes for `rs_gguf_loader_core` are recorded here.

## [Unreleased]

### Added

- Initial crate scaffold: GGUF v3 header parsing (`parse_gguf_header_bytes`), magic and version validation, `GgufHeaderV3` with `estimated_index_bytes()` planning helper, `GgufLoadPlan` struct for prefetch/mmap/pinned-staging configuration, and `GgufError` typed error variants.
