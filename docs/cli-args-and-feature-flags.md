# CLI Args and Feature Flags (Current)

This document captures active CLI/runtime controls for current donor-harvest and Wave-1 integration tooling.

## 1) Donor/WIP Queue Executor

Script:

- `/home/local/ai/projects/scripts/apply_wip_kernel_queue.py`

### Args

- `--queue <path>` queue JSON path
- `--top <n>` process first N entries (`0` = all)
- `--target-surface <name>` filter by target surface
- `--category <name>` filter by category
- `--rust-priority <high|medium|native_kernel>` filter by rust priority
- `--only-cpu-audio` include only CPU-audio candidates
- `--preserve-priority <high|medium|low>` donor preserve priority filter
- `--repo-pattern <regex>` include repos by regex
- `--path-pattern <regex>` include paths by regex
- `--exclude-path-pattern <regex>` exclude paths by regex
- `--apply` execute copies (default is dry-run)
- `--overwrite` overwrite existing destination files
- `--summary-out <path>` output summary JSON

## 2) Wave-1 KV Policy Canary Runner

Script:

- `/home/local/ai/projects/gfxATOM-Rust/scripts/wave1_kv_policy_canary.py`

### Args

- `--policy-family <baseline|ratequant|deltak|wobble|qaq|nautilus>`
- `--policy-mode <strict|fallback>`
- `--enable-donor-features`
- `--enable-ratequant`
- `--enable-deltak`
- `--enable-wobble`
- `--enable-qaq`
- `--enable-nautilus`
- `--out <path>` output decision JSON
- `--total-budget <float>` ratequant budget simulation
- `--sensitivities <csv>` per-head sensitivity simulation
- `--vector-dim <int>` turboquant vector dimension
- `--bits-per-value <int>` requested bits before guardrail
- `--outlier-ratio <float>` norm-guardrail outlier ratio
- `--adaptive` include wave-2 adaptive family recommendation
- `--kv-hit-rate <float>`
- `--prefix-reuse-ratio <float>`
- `--prefill-tps <float>`
- `--decode-tps <float>`
- `--storage-tier <hbm|cpu|nvme|object>`

### Example

```bash
python /home/local/ai/projects/gfxATOM-Rust/scripts/wave1_kv_policy_canary.py \
  --policy-family ratequant \
  --policy-mode strict \
  --enable-donor-features \
  --enable-ratequant \
  --out /tmp/wave1_kv_ratequant_canary.json
```

## 3) Runtime Environment Feature Flags

Global:

- `GFXATOM_DONOR_FEATURES=0|1`
- `GFXATOM_KV_POLICY_MODE=strict|fallback`
- `GFXATOM_KV_POLICY_FAMILY=baseline|ratequant|deltak|wobble|qaq|nautilus|rkv`

Per-feature:

- `GFXATOM_KV_RATEQUANT=0|1`
- `GFXATOM_KV_DELTAK=0|1`
- `GFXATOM_KV_WOBBLE=0|1`
- `GFXATOM_KV_QAQ=0|1`
- `GFXATOM_KV_NAUTILUS=0|1`
- `GFXATOM_KV_RKV=0|1`

## 4) Fail-Closed Behavior

If requested policy is unknown, disabled, or globally gated off:

- selected policy falls back to `baseline`
- decision is marked `accepted=false`
- rejection reason is populated in canary profile output
- adaptive recommendation is informational and does not override requested policy selection
