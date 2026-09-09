# Benchmark provenance

The caches behind `bench/gen_table*.md`, `docs/src/assets/perf_*.svg` and the generated tables in `docs/src/coverage.md`. Both reference views (OpenBLAS, AOCL) are rendered from this one cache set. Methodology: `docs/src/methodology.md`.

| µarch | CPU | commit | measured |
|---|---|---|---|
| Zen3 · AVX2 | AMD Ryzen 9 5900X 12-Core Processor | `5647c65` | 2026-09-09T10:40 |
| Zen4 · AVX-512 | AMD Ryzen 5 7640U w/ Radeon 760M Graphics | `5647c657` | 2026-09-09T10:58 |
| Zen5 · AVX-512 | AMD Ryzen AI 5 340 w/ Radeon 840M | `5647c65` | 2026-09-09T10:41 |
