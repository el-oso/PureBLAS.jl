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

**Tier:** 62 Derived · 10 Measured · 40 Literal · 8 Exempt.
**Default form** (mechanical): 54 formula · 20 delegates · 6 sibling · 35 literal · 4 flag · 1 other.


## BLAS-1 SIMD kernels

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `axpy_dram` | formula | Derived | narrow 256-bit arm iff the datapath double-pumps; +17% Zen4, loses on Zen3. | n/a |
| `axpy_unroll` | formula | Derived | formula over detected consts: `_at_axpy_band(_HW)` | — |
| `zaxpy_narrow` | delegates | Derived | the criterion IS the datapath: narrow arm iff _datapath_bytes <= 32. | n/a |

## BLAS-2 (gemv/ger/trmv/trsv)

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `cgemv_mr` | literal | Literal | parameterises `_gemv_n_cmplx!`, which gemv-N NO LONGER USES (routed to _gemv_n_ri_cmplx!); reached only via hemv. Re-measure through zhemv, not zgemvN. | candidate, hemv shapes |
| `cgemv_rb` | formula | Derived | L2/16: the m*n threshold where a resident panel stops beating a stream. | n/a |
| `cgemvn_nc` | literal | Literal | 4 columns per panel, matching what OpenBLAS uses; not swept independently. | candidate |
| `cgemvn_nc_big` | formula | Derived | formula over detected consts: `3 * _vwidth(Float64) ÷ 2` | — |
| `cgemvn_pf` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4` | — |
| `cgemvt_cfg` | sibling | Derived | _CGEMVT_CFG_BIG is already a complete _NVREG derivation. | n/a |
| `cgemvt_half` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4` | — |
| `cgemvt_nc` | literal | Exempt | legacy pin, superseded by _cgemvt_cfg; retained only so an old preference still parses. | n/a, dead |
| `cgemvt_pf` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4` | — |
| `gemvn_mb` | formula | Derived | formula over detected consts: `max(_vwidth(Float64), _L1_BYTES ÷ 2 ÷ sizeof(Float64` | — |
| `gemvn_minner` | delegates | Measured | panel-width regime inverts across µarchs (Zen3/Zen4 keep the gain, Zen5 reverts). | candidate |
| `gemvn_minner_maxa` | formula | Derived | formula over detected consts: `4 * _L3_BYTES` | — |
| `gemvn_np_narrow` | formula | Derived | formula over detected consts: `max(2, _L1D_ASSOC - 2` | — |
| `gemvn_rb` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4 ? 64 : 448` | — |
| `gemvt_deep` | delegates | Derived | NC=8 x U=4 gated on A <= L2 and the register budget; both detected consts. | n/a |
| `gemvt_nc` | delegates | Literal | _ILP_TARGET predicts 8, every box measures 4; the derivation is falsified. | no, µarch-invariant |
| `gemvt_percol_amin` | formula | Measured | window's lower edge; no formula places it (see gemvt_perscan). | candidate, cache-boundary set |
| `gemvt_percol_xmax` | formula | Measured | window's upper edge, and the one worth money (Zen5 wants 4 KiB, not L1/2). | candidate |
| `gemvt_perscan` | delegates | Measured | every pair of boxes disagrees at some size; L1, L2, L3 and width all falsified. | not yet |
| `gemvt_pf` | delegates | Measured | prefetch distance depends on L2 hit latency and the hw streamer, neither detected; bounds derived, choice measured. | candidate, 0/2/4/8 lines |
| `gemvt_u` | delegates | Derived | row unroll capped by the register file: NC*U + U + 2 <= _NVREG. | n/a |
| `ger_panel_np` | delegates | Measured | optimum 8/4/1 on boxes that agree on L2/L3/width; tracks DRAM write streams. | 22 s |
| `tri_c_blk_min` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4 ? 256 : 1024` | — |
| `tri_c_t_unb` | literal | Literal | complex transpose unblocked/blocked crossover. | candidate |
| `tri_nb` | formula | Derived | formula over detected consts: `clamp(_round_dn(isqrt(_L1_BYTES ÷ 8), 16), 16, 64` | — |
| `tri_t_unb` | literal | Literal | unblocked/blocked crossover for the transpose path; blocking fixes a measured regression above it. | candidate |
| `trmv_f_dram` | literal | Derived | majority criterion: switch once tri > 2*L3. The 2 IS the 1/2. | n/a |
| `trmv_f_switch` | literal | Derived | NOT Measure-tier debt, despite the label this line carried until 2026-08-21. It is a MAJORITY CRITERION over a derived quantity: switch to the narrow panel once more than half the triangle's stream is DRAM-served, i.e. `1 - L3/tri > 1/2` <=> `tri > 2*L3`. The 2 IS the 1/2 — it is not a tuned multiplier, and the cache term carries the hardware. Validated at the boundary: Zen3 n=4096 sits exactly AT 2*L3 and measured 0.973 either way, so the switch costs nothing where it fires. | n/a — Derived |
| `trmv_fused_min` | delegates | Measured | a crossover set by call overhead vs vectorised work; sweepable without editing code. | candidate |
| `trsv_reg_max` | formula | Derived | formula over detected consts: `_SCALAR_FPREGS - 4` | — |
| `zhemv_pf` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4` | — |
| `zhemv_pf_tiles` | literal | Literal | prefetch depth in tiles for the Hermitian mat-vec. | candidate |

## BLAS-2 banded

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `gbmv_conv_max` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4 ? 20 : 48` | — |

## BLAS-2 packed

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `spmv_panel` | flag | Exempt | boolean switch (path on/off), not a tuned size. | — |
| `spmv_panel_minap` | formula | Derived | formula over detected consts: `_L1_BYTES` | — |

## BLAS-3 (trmm/trsm/syrk/symm)

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `chemm_pack_cut` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4 ? 4096 : 32` | — |
| `csymm_pack_cut` | sibling | Derived | same packed kernel as chemm, differing only by conjugation. | n/a, follows chemm |
| `csyr2k_fused_max` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4 ? 192 : 0` | — |
| `csyr2k_pack_cut` | literal | Literal | never swept: the retired ternary had two identical arms. Validated by gate only. | unswept |
| `csyrk_3m_min` | literal | Literal | lower bound for the 3M path; below it the Karatsuba overhead dominates. | candidate |
| `csyrk_pack_cut` | literal | Literal | trans='N': recurse below this rather than pack. | candidate |
| `csyrk_pack_cut_t` | literal | Literal | trans='C'/'T': packing wins almost always, hence the much lower cut. | candidate |
| `csyrk_unified_max` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4 ? 512 : 0` | — |
| `csyrk_unpack_max` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4 ? 16 : 192` | — |
| `ctrmm_pack` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4` | — |
| `ctrmm_pack_min` | literal | Literal | complex trmm pack threshold. | candidate |
| `ctrsm_direct_max` | literal | Literal | trtri overhead plus extra flops sink small/mid n, so the direct path stops here. | candidate |
| `ctrsm_ncut` | literal | Literal | B-width cut: at or below it, the j-outer narrow recursion wins. | candidate |
| `ctrsm_rec_l` | literal | Literal | recursion cut for complex trsm side-L; per-box. | candidate |
| `gemmtrsm_mr` | formula | Derived | formula over detected consts: `min(8, (_GT_NREG - _GT_NRV - 2) ÷ _GT_NRV` | — |
| `gemmtrsm_nrv` | formula | Derived | formula over detected consts: `_GT_NREG >= 32 ? 3 : 2` | — |
| `symm_pack_cut` | formula | Derived | formula over detected consts: `_at_symm_mat_max(_HW)` | — |
| `syr2k_2pass` | formula | Literal | AVX2-ONLY by construction: the default is typemax(Int) on AVX-512, which disables the branch. Zen3-only evidence is COMPLETE. | n/a off AVX2 |
| `syr2k_mr` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4 ? 2 : _MR` | — |
| `syr2k_nr` | sibling | Literal | drives its own microkernel, borrows gemm's _NR as a prior; unvalidated here. | candidate |
| `syr2k_pack_cut` | formula | Derived | formula over detected consts: `_at_rank_k_pack_cut(_HW)` | — |
| `syrk_dbase` | literal | Literal | diagonal-block base; larger pushes work into efficient off-diagonal gemms. | candidate |
| `syrk_mr` | literal | Literal | AVX2-ONLY by construction: `_tri_mr(T) = _vwidth(T)==4 ? _SYRK_MR : _MR`, so AVX-512 uses gemm's derived _MR. Zen3-only evidence is COMPLETE, not a gap. | n/a off AVX2 |
| `syrk_pack_cut` | formula | Derived | formula over detected consts: `_at_rank_k_pack_cut(_HW)` | — |
| `syrk_unified_max` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4 ? 48 : 0` | — |
| `trmm_ddirect` | literal | Literal | wide-SIMD-safe default for the direct path; per-box override without a code push. | candidate |
| `trmm_pack_min` | other | Derived | 5/2 x _GEMM_UNPACK_MAX, i.e. it follows gemm's own unpack bound. | n/a, follows gemm |
| `trmm_rkc` | sibling | Literal | own k-block, borrows gemm's _KC; a triangular operand packs differently. | candidate |
| `trmm_rpack` | literal | Literal | measured pack threshold; a box that disagrees pins it rather than editing. | candidate |
| `trsm_narrow_max` | literal | Literal | B-width below which the narrow path wins; measured, not derived. | candidate |
| `ztrsm_gt_mr` | formula | Derived | formula over detected consts: `_ZGT_W` | — |

## CPU detection

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `force_hooks` | flag | Exempt | boolean switch (path on/off), not a tuned size. | — |
| `simd_bytes` | delegates | Exempt | the detected SIMD width itself; the override exists for cross-compile and trim builds, not tuning. | n/a |

## LAPACK · banded_chol

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `pbtrf_cross_kd` | delegates | Literal | a crossover, not a width; lanes- and byte-budget derivations both falsified. | candidate |
| `pbtrf_nb` | delegates | Derived | a panel width is counted in vector registers. | n/a |
| `pbtrf_nb_small` | delegates | Derived | same unit; the F32 method is the pure formula _lanes(hw, Float32). | n/a |
| `pbtrf_u_native_kd` | delegates | Literal | crossover; F64 breaks the l2/4096 rule its F32 sibling obeys. | candidate |

## LAPACK · bunchkaufman

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `sytrf_cmult` | delegates | Measured | the complex multiplier is the Measure-tier knob here; gate-measured 3, the duel never picked it. | candidate |
| `sytrf_nb` | delegates | Derived | panel width from the shape function; only the multiplier below it is measured. | n/a |

## LAPACK · gbtrf

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `gbtrf_cross` | delegates | Literal | crossover, derivation falsified; the C32 row is a 3-3 tie on Zen4. | candidate |
| `gbtrf_nb` | delegates | Measured | banded LU panel width; the comment says outright it needs measuring, not assuming. | candidate |

## LAPACK · lapack

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `chol_base_split` | formula | Derived | formula over detected consts: `_INTEL_AVX2` | — |
| `chol_nb` | literal | Literal | trsm panel width, measured µarch-invariant; confirm on the fleet before deriving. | candidate |
| `chol_nc` | literal | Literal | syrk column block, same status as chol_nb. | candidate |
| `cpotf2_mr` | formula | Derived | formula over detected consts: `_at_cpotf2_mr(_HW)` | — |
| `cpotrf_base` | formula | Derived | 32 + 4*lanes: a width-independent overhead floor plus a per-lane slope. | n/a |
| `cpotrf_nbmax` | formula | Derived | formula over detected consts: `_at_cpotrf_nbmax(_HW)` | — |
| `potrf_base` | literal | Literal | recursion-overhead floor, µarch-flat at 16-32; the residency guess 64 is worse. | no |
| `potrf_base_f32` | sibling | Derived | sizeof ratio: F32 is half F64's bytes, so half the n. | n/a, follows potrf_base |
| `potrf_pad` | flag | Exempt | boolean switch (path on/off), not a tuned size. | — |
| `potrf_upper_direct_max` | delegates | Measured | a tiny-n crossover inside a noisy band; only host measurement resolves it. | candidate |

## LAPACK · packed_chol

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `pptrf_blk_min` | literal | Literal | smallest n where unpacking pays; measured 2.26x at n=32, untested between 8 and 32. | candidate |
| `pptrf_blk_nb` | sibling | Literal | own panel width, borrows _LU_NB, which is itself a falsified derivation. | candidate |
| `pptrf_spr_min` | literal | Exempt | 0 is the unset sentinel, not a size. | n/a |

## LAPACK · pstrf

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `pstrf_fuse_max` | formula | Derived | formula over detected consts: `(_L1_BYTES ÷ 2) ÷ _CACHELINE` | — |
| `pstrf_rowcache_min` | literal | Literal | measured on Zen4 F64 ONLY; not yet fleet-validated, so not a derivation. | candidate |

## LAPACK · qr

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `qr_nb_c` | literal | Literal | complex QR panel; the Zen4 sweet spot, keyed per box via Preferences. | candidate |

## LAPACK · svd

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `brd_nb` | delegates | Literal | machine-INVARIANT 8 on three boxes and two ISAs; a formula taking hw and ignoring it was removed. | no |
| `bt_nb` | literal | Literal | back-transform block, measured invariant; confirm on a very-wide-register box before deriving. | candidate |

## gemm (BLAS-3)

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `cgemm_3m` | flag | Exempt | boolean switch (path on/off), not a tuned size. | — |
| `cgemm_3m_kmin` | literal | Literal | 3M needs min(m,n,k) >= this or the Karatsuba overhead dominates a thin gemm. | candidate |
| `cgemm_3m_max` | literal | Literal | upper edge of the 3M window; above it the extra passes cost more than the flop cut. | candidate |
| `cgemm_3m_min` | literal | Literal | lower edge of the 3M window on max(m,n,k). | candidate |
| `cgemm_kc` | formula | Derived | formula over detected consts: `_l1_block(_HW, ComplexF64, max(_CNR, _CNR_SMALL` | — |
| `cgemm_mr` | formula | Derived | formula over detected consts: `_W64 == 4 ? 1 : 2` | — |
| `cgemm_nr` | formula | Derived | formula over detected consts: `_W64 == 4 ? 6 : 4` | — |
| `cgemm_nr_small` | formula | Derived | formula over detected consts: `_W64 == 4 ? 4 : _CNR` | — |
| `cgemm_nrsmall_max` | formula | Derived | formula over detected consts: `_W64 == 4 ? 64 : 0` | — |
| `cgemm_tiny` | literal | Literal | tiny-n bypass for complex gemm. | candidate |
| `cgemm_unpack_max` | formula | Derived | formula over detected consts: `_W64 == 4 ? 40 : 192` | — |
| `cuker_nr6_min` | formula | Derived | formula over detected consts: `_W64 == 4 ? 48 : typemax(Int` | — |
| `gemm_kc` | formula | Derived | formula over detected consts: `_at_gemm_kc(_HW)` | — |
| `gemm_mr` | formula | Derived | formula over detected consts: `_at_gemm_mr(_HW)` | — |
| `gemm_mr1_max` | formula | Derived | formula over detected consts: `_at_gemm_mr1_max(_HW)` | — |
| `gemm_nc` | formula | Derived | formula over detected consts: `_at_gemm_nc(_HW)` | — |
| `gemm_nr` | formula | Derived | formula over detected consts: `_at_gemm_nr(_HW)` | — |
| `gemm_split_max` | formula | Derived | formula over detected consts: `_at_gemm_split_max(_HW)` | — |
| `gemm_unpack_max` | formula | Derived | formula over detected consts: `_at_gemm_unpack_max(_HW)` | — |
| `strassen` | formula | Exempt | capability flag; Strassen's flop cut is ISA-independent. | n/a |
| `strassen_maxdepth` | literal | Literal | recursion depth cap; deeper trades flops for pack/add traffic. | candidate |
| `strassen_min` | literal | Literal | split while min(m,n,k) >= this; measured, and the base stays >= ~min/2. | candidate |

## workspace

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `l3_nb` | formula | Derived | formula over detected consts: `clamp(_round_dn(isqrt(_L2_BYTES ÷ 32), 16), 16, 128` | — |

---

Const names, defaults and files are deliberately NOT tabulated: they are one `grep` away
and made this table too wide to read. The knob key is the identifier that matters.

## Tuning constants that are NOT knobs

34 `const _X = <literal>` values in `src/` with no `@load_preference`.
They are tuning constants all the same — and in a WORSE position than a knob, because
they cannot be pinned, cannot be tuned by `tune!()`, and were invisible to the audit
above. `trtrs` is the worked example: its real path (trsm side-L) runs almost entirely
on these, not on knobs.

**Tier:** 9 Derived · 23 Literal · 2 Exempt.


### BLAS-1 SIMD kernels

| Const | Value | Tier | Why |
|---|---|---|---|
| `_IAMAX_NB_STREAM` | 4 | Derived | ILP chain count, ISA-invariant by the same latency x throughput argument. |
| `_UNROLL` | 4 | Derived | 4 independent chains x W lanes, an ILP count tied to _ILP_TARGET. |

### BLAS-2 (gemv/ger/trmv/trsv)

| Const | Value | Tier | Why |
|---|---|---|---|
| `_CGEMV_NP` | 8 | Derived | complex gemv-N panel width, same register-pressure argument as _GEMV_NP. DERIVABLE. |
| `_GEMVN_MINNER_U` | 4 | Literal | row-unroll paired with the gemvn_minner knob; moves with it, not independently. |
| `_GEMVT_NC_DEEP` | 8 | Derived | one x-load per 8 FMAs, the load:FMA ratio that clears the MLP plateau; guarded by _NVREG. |
| `_GEMVT_U_DEEP` | 4 | Derived | 4 lines per stream, so NC*U = 32 lines named: enough to cover L2. |
| `_GEMV_NP` | 8 | Derived | gemv-N panel width; the comment already reasons in MR and register pressure. DERIVABLE from _NVREG. |
| `_GER_PANEL_U` | 4 | Literal | its own comment calls it 'a genuine tuning knob'. TUNABLE, and never made one. |
| `_SCALAR_FPREGS` | 16 | Derived | x86 has 16 scalar FP registers, AArch64 32; exactly the _NVREG pattern. DERIVABLE, currently hardcoded. |
| `_SYMV_MR` | 4 | Derived | register-file bound (its comment: 16 ymm, and the gemv-N MR=8 bump SPILLED symv). DERIVABLE from _NVREG. |
| `_SYMV_NB` | 8 | Derived | symv panel width, a lanes multiple. DERIVABLE from _lanes. |
| `_TRSV_T_F` | 8 | Literal | trsv-T fuse factor; the routing bound is expressed as a multiple of it. TUNABLE. |

### BLAS-3 (trmm/trsm/syrk/symm)

| Const | Value | Tier | Why |
|---|---|---|---|
| `_SYRK_BASE` | 48 | Literal | syrk recursion base before the off-diagonal gemm. TUNABLE. |
| `_TRMM_RPANEL` | 512 | Literal | trmm side-R panel width. TUNABLE, should be a knob. |
| `_TRSM_BASE` | 32 | Literal | trsm recursion base. TUNABLE and not a knob: unpinnable, untunable, and on trtrs's real path. |
| `_TRSM_DBASE` | 32 | Literal | diagonal-block base; its own comment says 'could be a Preference'. TUNABLE. |
| `_TRSM_NCUT` | 64 | Literal | B-width cut for side-L. TUNABLE, should be a knob. |
| `_TRSM_NCUT_R` | 128 | Literal | B-width cut for side-R. TUNABLE, should be a knob. |
| `_TRSM_R_FUSE` | 128 | Literal | side-R fuse threshold. TUNABLE, should be a knob. |
| `_TRTRI_BASE` | 16 | Literal | triangular-inverse recursion base. TUNABLE, should be a knob. |

### CPU detection

| Const | Value | Tier | Why |
|---|---|---|---|
| `_GEMM_SPLIT_S` | 2 | Literal | split-reduction factor, measured crossover ~n=56 on Zen4. TUNABLE. |
| `_ILP_TARGET` | 16 | Literal | 2 x FMA latency(4) x FMA ports(2); an architectural assumption, and neither latency nor port count is detectable. |
| `_TUNE_NKNOBS` | 12 | Exempt | a COUNT of Measure-tier knobs (the Bonferroni denominator), not a tuning value. |

### LAPACK · eigen_dc

| Const | Value | Tier | Why |
|---|---|---|---|
| `_STEDC_NB` | 25 | Literal | LAPACK's SMLSIZ: algorithm-intrinsic, machine-independent. tuning.md §4. |

### LAPACK · lapack

| Const | Value | Tier | Why |
|---|---|---|---|
| `_TR_TB` | 32 | Literal | residency-INVARIANT: two 32^2 F64 tiles = 16 KB, under any real L1. Deriving it would change nothing. |

### LAPACK · lu

| Const | Value | Tier | Why |
|---|---|---|---|
| `_GETF2_BASE` | 16 | Literal | store-traffic algorithm switch, invariant of cache size. tuning.md §4. |
| `_LU_NB` | 48 | Literal | residency derivation FLEET-FALSIFIED; the curve is parity-bumpy, not residency-shaped. tuning.md §4. |

### LAPACK · packed_chol

| Const | Value | Tier | Why |
|---|---|---|---|
| `_PPTRF_TPSV_MIN` | 32 | Literal | call-overhead vs vectorised-work crossover, measured Zen4 ONLY; needs fleet validation. |

### LAPACK · qr

| Const | Value | Tier | Why |
|---|---|---|---|
| `_QR_UNBLK_MAX` | 32 | Literal | unblocked/blocked crossover, unmeasured off Zen4; pinned flat deliberately. tuning.md §4. |

### LAPACK · svd

| Const | Value | Tier | Why |
|---|---|---|---|
| `_BRD_NB` | 8 | Literal | machine-INVARIANT 8 on three boxes and two ISAs. |
| `_SVD_DC_CROSS` | 1 | Exempt | a CORRECTNESS override, not a perf choice: bdsqr fails on clustered sigma, so all with-vectors SVD routes to D&C. |

### LAPACK · svd_dc

| Const | Value | Tier | Why |
|---|---|---|---|
| `_DC_THRESHOLD` | 64 | Literal | divide-and-conquer cut. TUNABLE, algorithm-intrinsic so no formula. |
| `_SEC_BISECT_CAP` | 0 | Literal | secular-equation bisection cap; 0 disables. TUNABLE. |

### gemm (BLAS-3)

| Const | Value | Tier | Why |
|---|---|---|---|
| `_GEMM_TINY` | 6 | Literal | below this the naive loop beats the packed path. TUNABLE. |
