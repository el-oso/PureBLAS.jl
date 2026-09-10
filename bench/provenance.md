# Benchmark provenance

The caches behind `bench/gen_table*.md`, `docs/src/assets/perf_*.svg` and the generated tables in `docs/src/coverage.md`. Both reference views (OpenBLAS, AOCL) are rendered from this one cache set. Methodology: `docs/src/methodology.md`.

| µarch | CPU | commit | measured |
|---|---|---|---|
| Zen3 · AVX2 | AMD Ryzen 9 5900X 12-Core Processor | `69fcbc8` | 2026-09-10T10:54 |
| Zen4 · AVX-512 | AMD Ryzen 5 7640U w/ Radeon 760M Graphics | `9d98b6bf` | 2026-09-10T06:10 |
| Zen5 · AVX-512 | AMD Ryzen AI 5 340 w/ Radeon 840M | `69fcbc8` | 2026-09-10T09:49 |
