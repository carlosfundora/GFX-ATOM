# Wave-1 Feature: Norm-Separated Guardrail

## Source donor extraction

- Donor: `norm-separated-quantization`
- Extracted implementation paths:
  - `src/nsq.py`
  - `src/quant.py`
  - `src/datautils.py`

## Assimilation target

- `gfxATOM-Rust/python/wave1_donor_adapters.py`
  - `norm_separated_guardrail(...)`

## Runtime gating

- Always available as a safety profile in wave-1 policy composition.
- Applied alongside selected policy profile outputs.

## Behavior

- For high outlier ratios, recommends split quant lane.
- Derives an adjusted bit recommendation to preserve stability.

## Fallback behavior

- Input validation enforces outlier ratio in `[0, 1]` and positive bit-width.

