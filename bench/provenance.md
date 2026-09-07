# Benchmark provenance

The caches behind `bench/gen_table*.md`, `docs/src/assets/perf_*.svg` and the generated tables in `docs/src/coverage.md`. Both reference views (OpenBLAS, AOCL) are rendered from this one cache set. Methodology: `docs/src/methodology.md`.

| µarch | CPU | commit | measured |
|---|---|---|---|
| Zen3 · AVX2 | AMD Ryzen 9 5900X 12-Core Processor | `eafb91c` | 2026-09-07T20:14 |
| Zen4 · AVX-512 | AMD Ryzen 5 7640U w/ Radeon 760M Graphics | `eafb91c` | 2026-09-07T20:25 |
| Zen5 · AVX-512 | AMD Ryzen AI 5 340 w/ Radeon 840M | `eafb91c` | 2026-09-07T20:21 |
