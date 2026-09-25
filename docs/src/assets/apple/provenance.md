# Benchmark provenance

The caches behind `bench/gen_table*.md`, `docs/src/assets/perf_*.svg` and the generated tables in `docs/src/coverage.md`. Both reference views (OpenBLAS, AOCL) are rendered from this one cache set. Methodology: `docs/src/methodology.md`.

| µarch | CPU | commit | measured |
|---|---|---|---|
| ARM · NEON | Apple M6 | `e8932f1e` | 2026-09-24T21:10 |
