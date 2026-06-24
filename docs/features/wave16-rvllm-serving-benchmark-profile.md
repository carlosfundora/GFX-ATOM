# Wave-16 rvLLM serving benchmark profile

rvLLM was assimilated as a serving/benchmark capability profile rather than as a GPU or TPU backend port.

## Captured capability surface

- Rust-native GPU serving path
- pure JAX + XLA TPU serving path
- cross-device benchmarking
- single graph capture
- FP8 channelscale epilogue
- cached TTFT reporting
- peak throughput reporting
- perplexity reporting
- streaming API server support

## Integration result

- Engine runtime profiles in Python and Rust now expose serving and benchmark telemetry flags.
- The profile remains fail-closed and read-only; it does not add a new execution backend.
- The capability surface is covered by Python/Rust parity tests.
