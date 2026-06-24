# Wave-2 Feature: Adaptive Policy Recommendation Lane

## Purpose

Provide an advisory policy recommendation from runtime signals without overriding explicit operator policy selection.

## Assimilation target

- `gfxATOM-Rust/python/wave2_adaptive_policy.py`
  - `RuntimeSignals`
  - `AdaptivePolicyRecommendation`
  - `recommend_policy_family(...)`

## CLI surface

- `gfxATOM-Rust/scripts/wave1_kv_policy_canary.py`
  - `--adaptive`
  - `--kv-hit-rate`
  - `--prefix-reuse-ratio`
  - `--prefill-tps`
  - `--decode-tps`
  - `--storage-tier`

## Heuristics

- `ratequant` for high prefix reuse + high KV hit rate.
- `nautilus` for high outlier ratio.
- `deltak` for cold storage tiers with low KV hit rate.
- `wobble` when decode is heavily skewed versus prefill and no stronger recommendation applies.
- fallback: `baseline`.

## Safety boundary

- Recommendation is output-only metadata.
- Current canary path does not mutate selected policy from this recommendation.

