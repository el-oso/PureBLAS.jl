# Benchmark provenance

The caches behind `bench/gen_table*.md`, `docs/src/assets/perf_*.svg` and the generated tables in `docs/src/coverage.md`. The reference views (OpenBLAS, Accelerate) are rendered from this one cache set. Methodology: `docs/src/methodology.md`.

| µarch | CPU | commit | measured |
|---|---|---|---|
| ARM · NEON | Apple M6 | `11f9f346d` | 2026-11-16T23:46 |
