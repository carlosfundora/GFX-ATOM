# Changelog

## Unreleased

- Added local-first docs for architecture, provenance, runtime contracts, benchmarks, and testing policy.
- Defined anti-mock rules for benchmark and acceptance paths.
- Added `benchmarks/` and `evidence/` output conventions for reproducible runs.
- Added `docs/rust-target.md` and linked it from project docs.
- Pinned donor/model index entries to explicit commit or revision identifiers.
- Added policy lint automation (`scripts/policy_lint.py`) and wired it into CI.
- Flattened donor repos to local immutable copies by removing live `.git` metadata.
