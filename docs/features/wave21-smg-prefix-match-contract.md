# Wave-21 SMG prefix-match contract

SMG’s prefix-match result contract was assimilated into the shared KV contract layer.

## Captured contract surface

- tenant IDs as interned routing identities
- prefix-match result types
- match-count and input-count accessors
- hit-ratio helper

## Integration result

- `rs_kv_quant_contracts` now exposes a reusable prefix-match result type and match-result trait.
- `rs_kv_validation_harness` validates the hit-ratio contract on a representative match result.
- The contract remains backend-neutral and is suitable for future positional-index routing work.
