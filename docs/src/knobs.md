# The Knob Registry

!!! warning "Generated file — do not edit"
    Produced by `test/knob_registry.jl` from `src/`. To change a row, change the code
    or its `# PDM:` marker and regenerate. `test/knob_registry_tests.jl` fails if this
    file is out of date.

Every `@load_preference` key in `src/` — 120 of them.

| Tier | Meaning |
|---|---|
| **Derived** | Default is a formula over detected hardware consts. |
| **Measured** | Not derivable — the optimum depends on something we cannot detect. Wants a `tune!()` pin. |
| **Literal** | A fixed value: a proven invariant, or a derivation that was tried and falsified. |
| **Exempt** | Not hardware tuning at all — a sentinel or a capability flag. |
| **Unaudited** | Nobody has classified it yet. Debt, not a verdict. |

**Counts:** 8 Derived · 4 Measured · 9 Literal · 2 Exempt · 97 Unaudited.

**23 / 120 classified.** An unaudited knob is one nobody has justified —
which is how a redundant or mislabelled knob survives. The list below is the worklist.


## BLAS-1 SIMD kernels

| Knob | Tier | Why | `tune!()` |
|---|---|---|---|
| `axpy_dram` | Derived | narrow 256-bit arm iff the datapath double-pumps; +17% Zen4, loses on Zen3. | n/a |
| `axpy_unroll` | Unaudited | — | — |
| `zaxpy_narrow` | Unaudited | — | — |

## BLAS-2 (gemv/ger/trmv/trsv)

| Knob | Tier | Why | `tune!()` |
|---|---|---|---|
| `cgemv_mr` | Unaudited | — | — |
| `cgemv_rb` | Unaudited | — | — |
| `cgemvn_nc` | Unaudited | — | — |
| `cgemvn_nc_big` | Unaudited | — | — |
| `cgemvn_pf` | Unaudited | — | — |
| `cgemvt_cfg` | Derived | _CGEMVT_CFG_BIG is already a complete _NVREG derivation. | n/a |
| `cgemvt_half` | Unaudited | — | — |
| `cgemvt_nc` | Unaudited | — | — |
| `cgemvt_pf` | Unaudited | — | — |
| `gemvn_mb` | Unaudited | — | — |
| `gemvn_minner` | Unaudited | — | — |
| `gemvn_minner_maxa` | Unaudited | — | — |
| `gemvn_np_narrow` | Unaudited | — | — |
| `gemvn_rb` | Unaudited | — | — |
| `gemvt_deep` | Unaudited | — | — |
| `gemvt_nc` | Literal | _ILP_TARGET predicts 8, every box measures 4; the derivation is falsified. | no, µarch-invariant |
| `gemvt_percol_amin` | Measured | window's lower edge; no formula places it (see gemvt_perscan). | candidate, cache-boundary set |
| `gemvt_percol_xmax` | Measured | window's upper edge, and the one worth money (Zen5 wants 4 KiB, not L1/2). | candidate |
| `gemvt_perscan` | Measured | every pair of boxes disagrees at some size; L1, L2, L3 and width all falsified. | not yet |
| `gemvt_pf` | Unaudited | — | — |
| `gemvt_u` | Unaudited | — | — |
| `ger_panel_np` | Measured | optimum 8/4/1 on boxes that agree on L2/L3/width; tracks DRAM write streams. | 22 s |
| `tri_c_blk_min` | Unaudited | — | — |
| `tri_c_t_unb` | Unaudited | — | — |
| `tri_nb` | Unaudited | — | — |
| `tri_t_unb` | Unaudited | — | — |
| `trmv_f_dram` | Derived | majority criterion: switch once tri > 2*L3. The 2 IS the 1/2. | n/a |
| `trmv_f_switch` | Derived | NOT Measure-tier debt, despite the label this line carried until 2026-08-21. It is a MAJORITY CRITERION over a derived quantity: switch to the narrow panel once more than half the triangle's stream is DRAM-served, i.e. `1 - L3/tri > 1/2` <=> `tri > 2*L3`. The 2 IS the 1/2 — it is not a tuned multiplier, and the cache term carries the hardware. Validated at the boundary: Zen3 n=4096 sits exactly AT 2*L3 and measured 0.973 either way, so the switch costs nothing where it fires. | n/a — Derived |
| `trmv_fused_min` | Unaudited | — | — |
| `trsv_reg_max` | Unaudited | — | — |
| `zhemv_pf` | Unaudited | — | — |
| `zhemv_pf_tiles` | Unaudited | — | — |

## BLAS-2 banded

| Knob | Tier | Why | `tune!()` |
|---|---|---|---|
| `gbmv_conv_max` | Unaudited | — | — |

## BLAS-2 packed

| Knob | Tier | Why | `tune!()` |
|---|---|---|---|
| `spmv_panel` | Unaudited | — | — |
| `spmv_panel_minap` | Unaudited | — | — |

## BLAS-3 (trmm/trsm/syrk/symm)

| Knob | Tier | Why | `tune!()` |
|---|---|---|---|
| `chemm_pack_cut` | Unaudited | — | — |
| `csymm_pack_cut` | Derived | same packed kernel as chemm, differing only by conjugation. | n/a, follows chemm |
| `csyr2k_fused_max` | Unaudited | — | — |
| `csyr2k_pack_cut` | Literal | never swept: the retired ternary had two identical arms. Validated by gate only. | unswept |
| `csyrk_3m_min` | Unaudited | — | — |
| `csyrk_pack_cut` | Unaudited | — | — |
| `csyrk_pack_cut_t` | Unaudited | — | — |
| `csyrk_unified_max` | Unaudited | — | — |
| `csyrk_unpack_max` | Unaudited | — | — |
| `ctrmm_pack` | Unaudited | — | — |
| `ctrmm_pack_min` | Unaudited | — | — |
| `ctrsm_direct_max` | Unaudited | — | — |
| `ctrsm_ncut` | Unaudited | — | — |
| `ctrsm_rec_l` | Unaudited | — | — |
| `gemmtrsm_mr` | Unaudited | — | — |
| `gemmtrsm_nrv` | Unaudited | — | — |
| `symm_pack_cut` | Unaudited | — | — |
| `syr2k_2pass` | Unaudited | — | — |
| `syr2k_mr` | Unaudited | — | — |
| `syr2k_nr` | Literal | drives its own microkernel, borrows gemm's _NR as a prior; unvalidated here. | candidate |
| `syr2k_pack_cut` | Unaudited | — | — |
| `syrk_dbase` | Unaudited | — | — |
| `syrk_mr` | Unaudited | — | — |
| `syrk_pack_cut` | Unaudited | — | — |
| `syrk_unified_max` | Unaudited | — | — |
| `trmm_ddirect` | Unaudited | — | — |
| `trmm_pack_min` | Unaudited | — | — |
| `trmm_rkc` | Literal | own k-block, borrows gemm's _KC; a triangular operand packs differently. | candidate |
| `trmm_rpack` | Unaudited | — | — |
| `trsm_narrow_max` | Unaudited | — | — |
| `ztrsm_gt_mr` | Unaudited | — | — |

## CPU detection

| Knob | Tier | Why | `tune!()` |
|---|---|---|---|
| `force_hooks` | Unaudited | — | — |
| `simd_bytes` | Unaudited | — | — |

## LAPACK · banded_chol

| Knob | Tier | Why | `tune!()` |
|---|---|---|---|
| `pbtrf_cross_kd` | Literal | a crossover, not a width; lanes- and byte-budget derivations both falsified. | candidate |
| `pbtrf_nb` | Derived | a panel width is counted in vector registers. | n/a |
| `pbtrf_nb_small` | Derived | same unit; the F32 method is the pure formula _lanes(hw, Float32). | n/a |
| `pbtrf_u_native_kd` | Literal | crossover; F64 breaks the l2/4096 rule its F32 sibling obeys. | candidate |

## LAPACK · bunchkaufman

| Knob | Tier | Why | `tune!()` |
|---|---|---|---|
| `sytrf_cmult` | Unaudited | — | — |
| `sytrf_nb` | Unaudited | — | — |

## LAPACK · gbtrf

| Knob | Tier | Why | `tune!()` |
|---|---|---|---|
| `gbtrf_cross` | Literal | crossover, derivation falsified; the C32 row is a 3-3 tie on Zen4. | candidate |
| `gbtrf_nb` | Unaudited | — | — |

## LAPACK · lapack

| Knob | Tier | Why | `tune!()` |
|---|---|---|---|
| `chol_base_split` | Unaudited | — | — |
| `chol_nb` | Unaudited | — | — |
| `chol_nc` | Unaudited | — | — |
| `cpotf2_mr` | Unaudited | — | — |
| `cpotrf_base` | Unaudited | — | — |
| `cpotrf_nbmax` | Unaudited | — | — |
| `potrf_base` | Literal | recursion-overhead floor, µarch-flat at 16-32; the residency guess 64 is worse. | no |
| `potrf_base_f32` | Derived | sizeof ratio: F32 is half F64's bytes, so half the n. | n/a, follows potrf_base |
| `potrf_pad` | Unaudited | — | — |
| `potrf_upper_direct_max` | Unaudited | — | — |

## LAPACK · packed_chol

| Knob | Tier | Why | `tune!()` |
|---|---|---|---|
| `pptrf_blk_min` | Unaudited | — | — |
| `pptrf_blk_nb` | Literal | own panel width, borrows _LU_NB, which is itself a falsified derivation. | candidate |
| `pptrf_spr_min` | Exempt | 0 is the unset sentinel, not a size. | n/a |

## LAPACK · pstrf

| Knob | Tier | Why | `tune!()` |
|---|---|---|---|
| `pstrf_fuse_max` | Unaudited | — | — |
| `pstrf_rowcache_min` | Unaudited | — | — |

## LAPACK · qr

| Knob | Tier | Why | `tune!()` |
|---|---|---|---|
| `qr_nb_c` | Unaudited | — | — |

## LAPACK · svd

| Knob | Tier | Why | `tune!()` |
|---|---|---|---|
| `brd_nb` | Unaudited | — | — |
| `bt_nb` | Unaudited | — | — |

## gemm (BLAS-3)

| Knob | Tier | Why | `tune!()` |
|---|---|---|---|
| `cgemm_3m` | Unaudited | — | — |
| `cgemm_3m_kmin` | Unaudited | — | — |
| `cgemm_3m_max` | Unaudited | — | — |
| `cgemm_3m_min` | Unaudited | — | — |
| `cgemm_kc` | Unaudited | — | — |
| `cgemm_mr` | Unaudited | — | — |
| `cgemm_nr` | Unaudited | — | — |
| `cgemm_nr_small` | Unaudited | — | — |
| `cgemm_nrsmall_max` | Unaudited | — | — |
| `cgemm_tiny` | Unaudited | — | — |
| `cgemm_unpack_max` | Unaudited | — | — |
| `cuker_nr6_min` | Unaudited | — | — |
| `gemm_kc` | Unaudited | — | — |
| `gemm_mr` | Unaudited | — | — |
| `gemm_mr1_max` | Unaudited | — | — |
| `gemm_nc` | Unaudited | — | — |
| `gemm_nr` | Unaudited | — | — |
| `gemm_split_max` | Unaudited | — | — |
| `gemm_unpack_max` | Unaudited | — | — |
| `strassen` | Exempt | capability flag; Strassen's flop cut is ISA-independent. | n/a |
| `strassen_maxdepth` | Unaudited | — | — |
| `strassen_min` | Unaudited | — | — |

## workspace

| Knob | Tier | Why | `tune!()` |
|---|---|---|---|
| `l3_nb` | Unaudited | — | — |

---

Const names, defaults and files are deliberately NOT tabulated: they are one `grep` away
and made this table too wide to read. The knob key is the identifier that matters.
