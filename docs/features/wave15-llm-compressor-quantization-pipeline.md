# Wave-15 LLM Compressor quantization pipeline

LLM Compressor was assimilated as a quantization-pipeline capability profile rather than as a new runtime backend.

## Captured capability surface

- model-free PTQ
- compressed-tensors output format
- weight quantization pipeline
- activation quantization pipeline
- KV cache quantization pipeline
- attention quantization pipeline
- disk-offloading quantization
- distributed calibration

## Integration result

- Engine runtime profiles in Python and Rust now expose quantization-pipeline capability flags.
- The profile remains fail-closed and read-only; it does not add a deployment backend or execution engine.
- The capability surface is covered by Python/Rust parity tests.
