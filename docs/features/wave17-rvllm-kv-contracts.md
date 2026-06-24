# Wave-17 rvLLM KV contract imports

rvLLM’s stronger Rust KV control-plane pieces were assimilated into the shared KV contract crate.

## Captured contract surface

- KV transfer plans
- KV prefetch plans
- KV eviction decisions
- agent step graphs
- precomputed context assets
- precomputed KV assets
- decode telemetry bundles

## Integration result

- The new contract types live in `rs_kv_quant_contracts`.
- Validation harness coverage exercises round-trip serialization for the imported shapes.
- These types stay backend-neutral and can be reused by policy, loader, or scheduler code without depending on rvLLM directly.
