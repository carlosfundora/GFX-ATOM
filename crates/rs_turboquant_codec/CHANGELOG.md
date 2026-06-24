# Changelog

All notable crate-specific changes for `rs_turboquant_codec` are recorded here.

## [Unreleased]

### Added

- Initial crate scaffold: two-stage KV compression with `PolarQuantizer`/`PolarCode` and `QjlQuantizer`/`QjlSketch` stages, `TurboQuantizer`/`TurboCode` top-level API, `BitWidth` enum (1–8 bit) with `max_loss()` accuracy floor, SIMD submodule, and optional Python FFI feature gate.
