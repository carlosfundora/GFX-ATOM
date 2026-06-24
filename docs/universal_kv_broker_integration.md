# Universal KV Broker Integration

## Overview

The Universal KV Broker is now integrated into gfxatom as a pluggable KV connector backend. This enables model-agnostic compressed KV cache ownership with support for TurboQuant and RotorQuant quantization modes.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Scheduler / Engine                        │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ├─ Selects KV connector backend via config
                 │
                 ▼
        ┌────────────────────────┐
        │ KVConnectorFactory     │
        │  (registry-based)      │
        └────────┬─────────────┬─┘
                 │             │
        ┌────────▼─┐    ┌──────▼──────────┐
        │ moriio   │    │universal_broker │  ◄─── NEW
        │ (RDMA)   │    │ (KV compression)│
        └──────────┘    └──────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Universal KV Broker │
                    │ - GPU KV pool       │
                    │ - RAM spill layer   │
                    │ - TurboQuant modes  │
                    │ - RotorQuant modes  │
                    │ - Observability     │
                    └─────────────────────┘
```

## Registration

The broker backend is automatically registered when `atom.kv_transfer.disaggregation.factory` is imported:

```python
KVConnectorFactory.register(
    "universal_broker",
    worker_module="atom.kv_transfer.universal_broker_adapter",
    worker_class="UniversalBrokerConnector",
    scheduler_module="atom.kv_transfer.universal_broker_adapter",
    scheduler_class="UniversalBrokerSchedulerConnector",
)
```

## Usage

### Enabling the broker

```bash
# Command-line flag
python -m atom.entrypoints.api_server \
  --model /path/to/model \
  --kv-connector universal_broker \
  --universal-kv-gpu-capacity-mb 2048 \
  --universal-kv-ram-capacity-mb 8192
```

### Configuration

```python
config = {
    "kv_transfer_config": {
        "kv_connector": "universal_broker",
        "gpu_capacity_mb": 2048,
        "ram_capacity_mb": 8192,
        "hot_importance_threshold": 0.7,
        "block_size": 16,
    }
}
```

## Supported KV Quantization Modes

The broker validates KV dtypes against its supported list:
- `fp16` — half-precision floating point
- `bf16` — bfloat16 floating point
- `fp8` — 8-bit floating point (with implicit dequant on access)
- `int8` — signed 8-bit integer (TurboQuant mode)
- `int4` — signed 4-bit integer (RotorQuant mode)

Unsupported modes trigger a warning and degrade to `fp16`.

## Observability

The scheduler-side connector tracks metrics:

```python
scheduler_conn.get_metrics()
# Returns:
# {
#     "allocations": 42,
#     "compressions": 156,
#     "spills": 12,
#     "evictions": 5,
#     "cache_hits": 1284,
#     "cache_misses": 321,
# }
```

Metrics can be disabled via `broker_config={"enable_metrics": False}`.

## Interface Compliance

### Worker-side (`UniversalBrokerConnector`)

Implements `KVConnectorBase`:
- `register_kv_caches(kv_caches: dict)` — Register local KV tensors for broker tracking
- `start_load_kv(metadata: ConnectorMetadata)` — Initiate async KV loads
- `get_finished() -> tuple[set, set]` — Report transfer completion status

### Scheduler-side (`UniversalBrokerSchedulerConnector`)

Implements `KVConnectorSchedulerBase`:
- `get_num_new_matched_tokens(seq)` — Check if sequence needs remote KV prefill
- `build_connector_meta() -> ConnectorMetadata` — Build transfer metadata snapshot
- `update_state_after_alloc(seq)` — Update state after block allocation
- `request_finished(seq)` — Handle request completion cleanup

## Key Design Decisions

### 1. Broker is independent from attention backend

The broker manages **where and how KV is stored** (compression, quantization, spill management). The attention backend (triton, wave, aiter, etc.) handles **how to compute** attention using that KV. They are orthogonal concerns:

```
Broker choices:     universal_broker  moriio  (other backends)
Attention choices:  triton  wave  aiter  flashinfer  (independent)

Any combination is valid.
```

### 2. Graceful fallback for unsupported modes

If a KV quantization mode is unsupported:
- Log a warning
- Degrade to the nearest supported dtype (typically `fp16`)
- Continue execution without failure

This ensures the system stays operational even with suboptimal KV compression.

### 3. Optional dependency

The broker requires `universal_kv` and `sglang` packages. If these are not available:
- Registration succeeds (backend is registered)
- Instantiation fails gracefully with a clear error message
- The system falls back to the default `moriio` backend

### 4. Observability first

Full-observability mode is enabled by default. Metrics collection includes:
- Allocation requests
- Compression operations
- Spill to RAM events
- Eviction events
- Cache hit/miss ratios

## Testing

Run the broker adapter tests:

```bash
cd /home/local/ai/engines/gfxatom
python -m pytest tests/test_universal_broker_adapter.py -v
```

Test coverage includes:
- Factory registration (✓)
- Connector instantiation (✓)
- Capability guards (✓)
- Metrics collection (✓)
- Interface compliance (✓)
- Factory integration (✓)

## Future Work

1. **Rust FFI for compression paths**: Replace Python compression/dequant with HIP/Triton kernels for gfx1030.
2. **Prefix reuse with broker**: Integrate radix cache prefix matching into broker block allocation.
3. **Quantized KV cache**: Extend broker to manage quantized block headers and implement dequant-on-access.
4. **Performance profiling**: Benchmark broker KV compression vs. standard paged KV on real workloads.
5. **Offline weight transformation**: Pre-transform model weights to broker-friendly layouts at model load time.

## References

- **Universal KV Broker** (SGLang donor): `/home/local/ai/projects/donors/sglang-1-bit-turbo/python/sglang/srt/layers/attention/universal_kv_broker.py`
- **Universal Broker Backend** (SGLang donor): `/home/local/ai/projects/donors/sglang-1-bit-turbo/python/sglang/srt/layers/attention/universal_broker_backend.py`
- **KV Transfer Types**: `/home/local/ai/engines/gfxatom/atom/kv_transfer/disaggregation/types.py`
- **KV Connector Factory**: `/home/local/ai/engines/gfxatom/atom/kv_transfer/disaggregation/factory.py`
- **Broker Adapter**: `/home/local/ai/engines/gfxatom/atom/kv_transfer/universal_broker_adapter.py`
- **Tests**: `/home/local/ai/engines/gfxatom/tests/test_universal_broker_adapter.py`
