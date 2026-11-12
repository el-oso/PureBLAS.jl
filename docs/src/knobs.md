# The Knob Registry

!!! warning "Generated file — do not edit"
    Produced by `test/knob_registry.jl` from `src/`. To change a row, change the code
    or its `# PDM:` marker and regenerate. `test/knob_registry_tests.jl` fails if this
    file is out of date.

Every `@load_preference` key in `src/` — 170 of them.

| Tier | Meaning |
|---|---|
| **Derived** | Default is a formula over detected hardware consts, or the detection itself. |
| **Measured** | Not derivable — the optimum depends on something we cannot detect. Wants a `tune!()` pin. |
| **Measured, orphaned** | Measured with no calibrator in `bench/calibrate.jl`, so `tune!()` can never set it and the pin it wants is unreachable. Its `tune!()` cell reads `none`. A missing tool, not a missing verdict — the tier says where the number came from and does not change when a calibrator arrives. |
| **Literal** | A fixed value: a proven invariant, or a derivation that was tried and falsified. |

**Tier:** 93 Derived · 24 Measured (15 calibrated · 9 orphaned) · 53 Literal.
**Default form** (mechanical): 70 formula · 24 delegates · 8 sibling · 61 literal · 7 other.


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
| `cgemvt_nc` | literal | Literal | column block of the L2-resident complex gemv-T/C regime; 4 beats 8 there (table in `_gemv_tc_cmplx!`), measured on Zen4 ONLY. | candidate |
| `cgemvt_pf` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4` | — |
| `cger_cold_den` | literal | Derived | DRAM residency of A with cache headroom, the `_trmv_blk!` criterion; fleet table above. | candidate |
| `gemv_mr` | delegates | Measured | the PANEL path's height; `_gemvn_rowblock_mr` below DERIVES the row-block path's height instead, and a pin wins for both tiers, disabling that derivation. | 0.3 s |
| `gemvn_mb` | formula | Derived | formula over detected consts: `max(_vwidth(Float64), _L1_BYTES ÷ 2 ÷ sizeof(Float64` | — |
| `gemvn_minner` | delegates | Derived | `_at_gemvn_minner(hw) = _datapath_bytes(hw) < 64`; the µarch split IS the datapath. | n/a — derived, no host measurement needed |
| `gemvn_minner_maxa` | formula | Derived | formula over detected consts: `4 * _L3_BYTES` | — |
| `gemvn_np_narrow` | formula | Derived | formula over detected consts: `max(2, _L1D_ASSOC - 2` | — |
| `gemvn_np_wide` | formula | Derived | formula over detected consts: register-capped `min(12, _NVREG - U - 1)` on a 32-register ISA, associativity-capped `_L1D_ASSOC` on 16 (the "wide" regime above) | — |
| `gemvn_rb` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4 ? 64 : 448` | — |
| `gemvt_deep` | delegates | Derived | NC=8 x U=4 gated on A <= L2 and the register budget; both detected consts. | n/a |
| `gemvt_nc` | delegates | Literal | _ILP_TARGET predicts 8, every box measures 4; the derivation is falsified. | no, µarch-invariant |
| `gemvt_percol_amin` | formula | Measured | window's lower edge; no formula places it (see gemvt_perscan). | candidate, cache-boundary set |
| `gemvt_percol_xmax` | formula | Measured | window's upper edge, and the one worth money (Zen5 wants 4 KiB, not L1/2). | candidate |
| `gemvt_perscan` | delegates | Measured | every pair of boxes disagrees at some size; L1, L2, L3 and width all falsified. | not yet |
| `gemvt_pf` | delegates | Measured | prefetch distance depends on L2 hit latency and the hw streamer, neither detected; bounds derived, choice measured. | candidate, 0/2/4/8 lines |
| `gemvt_u` | delegates | Derived | row unroll capped by the register file: NC*U + U + 2 <= _NVREG. | n/a |
| `ger_panel_np` | delegates | Measured | optimum 8/4/1 on boxes that agree on L2/L3/width; tracks DRAM write streams. | 22 s |
| `symv_sme_min` | sibling | Measured | where a single-pass symmetric kernel stops beating a two-pass split that reaches the matrix unit; the arms are structurally different and no cache size predicts the crossing. | none — orphaned; sweep n |
| `symv_sme_nb` | literal | Literal | a block width for a decomposition, not a hardware knob: it sets how many columns share one pass over x and y in the split, and 128 measured best or tied at every size swept. | candidate, (32,64,128,256) |
| `tri_c_blk_min` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4 ? 256 : 1024` | — |
| `tri_c_t_unb` | literal | Literal | complex transpose unblocked/blocked crossover. | candidate |
| `tri_nb` | formula | Derived | formula over detected consts: `clamp(_round_dn(isqrt(_L1_BYTES ÷ 8), 16), 16, 64` | — |
| `tri_t_unb` | literal | Literal | unblocked/blocked crossover for the transpose path; blocking fixes a measured regression above it. | candidate |
| `trmv_f_dram` | literal | Derived | majority criterion: switch once tri > 2*L3. The 2 IS the 1/2. | n/a |
| `trmv_f_switch` | literal | Derived | NOT Measure-tier debt, despite the label this line carried until 2026-08-21. It is a MAJORITY CRITERION over a derived quantity: switch to the narrow panel once more than half the triangle's stream is DRAM-served, i.e. `1 - L3/tri > 1/2` <=> `tri > 2*L3`. The 2 IS the 1/2 — it is not a tuned multiplier, and the cache term carries the hardware. Validated at the boundary: Zen3 n=4096 sits exactly AT 2*L3 and measured 0.973 either way, so the switch costs nothing where it fires. | n/a — Derived |
| `trmv_fused_min` | delegates | Literal | the L2-residency crossover was tried and falsified; fused8 wins at every n, all 3 boxes. | candidate |
| `trmv_sme_min` | sibling | Derived | the first n whose two-sweep cover carries a block worth an SME entry, 4*_TRI_NB, with four diagonal blocks beneath it; the arm above is `_trmv_split!`. Inert (typemax) without SME. | sweep n, upper/N |
| `trmv_split_recmax` | formula | Derived | formula over detected consts: `2 * _L2_BYTES` | — |
| `trsv_reg_max` | formula | Derived | formula over detected consts: `_SCALAR_FPREGS` | — |
| `zhemv_pf` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4` | — |
| `zhemv_pf_tiles` | literal | Literal | prefetch depth in tiles for the Hermitian mat-vec. | candidate |

## BLAS-2 banded

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `gbmv_conv_max` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4 ? 20 : 48` | — |
| `gbmv_mt_band_max` | formula | Derived | formula over `_vwidth`, physical criterion is the arm's `band/W` A-re-read factor; validated across band/W = 3.2..24.2 on AVX2, penalty 1.01-1.69x against a 2.8x threading gain. | candidate |

## BLAS-2 packed

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
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
| `syr2k_2pass` | formula | Literal | AVX2-ONLY by construction: the default is typemax(Int) on AVX-512, which disables the branch. Zen3-only evidence is COMPLETE. | n/a off AVX2 |
| `syr2k_mr` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4 ? 2 : _MR` | — |
| `syr2k_nr` | sibling | Literal | drives its own microkernel, borrows gemm's _NR as a prior; unvalidated here. | candidate |
| `syr2k_pack_cut` | formula | Derived | formula over detected consts: `_at_rank_k_pack_cut(_HW)` | — |
| `syr2k_sme_min` | literal | Derived | the first split's off-diagonal block must clear the tile-exact floor, 2 x _SME_MIN_EXACT; the same criterion as `_SYRK_SME_MIN`, with the table above as its falsification of the old 12*MR literal. | n/a, follows the tile |
| `syrk_base` | literal | Literal | syrk recursion base before the off-diagonal gemm. NOW A KNOB (was a bare const, unpinnable and untunable); default is the value it always had. | FLAT — 16..96 within noise on all 3 uarchs; largest cell +0.8% (Zen3 n=128) does not replicate (2026-08-21) |
| `syrk_dbase` | sibling | Derived | the leaf is bounded by the scratch the arena already reserves for it: `_L3_NB`. | n/a, follows the borrow |
| `syrk_mr` | literal | Literal | AVX2-ONLY by construction: `_tri_mr(T) = _vwidth(T)==4 ? _SYRK_MR : _MR`, so AVX-512 uses gemm's derived _MR. Zen3-only evidence is COMPLETE, not a gap. | n/a off AVX2 |
| `syrk_pack_cut` | formula | Derived | formula over detected consts: `_at_syrk_pack_cut(_HW)` | — |
| `syrk_sme_min` | literal | Derived | the first split's off-diagonal block must clear the tile-exact floor: 2 x _SME_MIN_EXACT. | n/a, follows the tile |
| `syrk_unified_max` | formula | Derived | formula over detected consts: `_vwidth(Float64) == 4 ? 48 : 0` | — |
| `trmm_base_r` | literal | Literal | recursion-overhead floor, flat 32-48, the same floor `_POTRF_BASE` carries; SME-only, the x86 side measured and regressing. | candidate |
| `trmm_ddirect` | literal | Literal | wide-SIMD-safe default for the direct path; per-box override without a code push. | candidate |
| `trmm_pack_min` | other | Derived | 5/2 x _GEMM_UNPACK_MAX, i.e. it follows gemm's own unpack bound. | n/a, follows gemm |
| `trmm_rkc` | sibling | Literal | own k-block, borrows gemm's _KC; a triangular operand packs differently. | candidate |
| `trmm_rpack` | formula | Literal | measured pack threshold; a box that disagrees pins it rather than editing. | candidate |
| `trmm_rpanel` | literal | Literal | trmm side-R panel width. NOW A KNOB (was a bare const, unpinnable and untunable); default is the value it always had. | FLAT — 64..1024 within noise on Zen3+Zen4+Zen5 except two non-replicating cells <=2.5% (2026-08-21) |
| `trsm_base` | literal | Literal | trsm recursion base, on trtrs's real path (trtrs wraps trsm side-L). NOW A KNOB (was a bare const, unpinnable and untunable); default is the value it always had. | FLAT — 16/32/48/64 all within noise on Zen3+Zen4+Zen5 (96 cells, 2026-08-21) |
| `trsm_dbase` | literal | Literal | diagonal-block base; its own comment already said 'could be a Preference'. NOW A KNOB (was a bare const, unpinnable and untunable); default is the value it always had. | FLAT — 16/32/48/64 within noise on Zen3+Zen4+Zen5 (2026-08-21) |
| `trsm_fullpack_min` | formula | Derived | formula over detected consts: `cld(2 * _L1_BYTES, _GT_NR * sizeof(Float64))`, the k at which the P-stripe outgrows 2·L1 (Zen4 measured 256<k≤384, formula 342; fleet validation still owed) | — |
| `trsm_fused_base` | formula | Derived | formula over detected consts on AVX-512: full-L1 residency of the KC×NR P-stripe, `_L1_BYTES ÷ (_GT_NR * sizeof)`; on SME the smallest block whose halving yields an `_sme_tile_ok`-eligible off-diagonal gemm, `4 * _SME_MR`; the remaining arm keeps a measured 128 (a bigger base regresses Zen3 n=256) | — |
| `trsm_narrow_max` | literal | Literal | B-width below which the narrow path wins; measured, not derived. | candidate |
| `trsm_ncut` | literal | Literal | B-width cut for side-L: at or below it the narrow recursion wins. NOW A KNOB (was a bare const, unpinnable and untunable); default is the value it always had. | FLAT — 32/64/128 within noise on Zen3+Zen4+Zen5 (2026-08-21) |
| `trsm_ncut_r` | literal | Literal | B-width cut for side-R. NOW A KNOB (was a bare const, unpinnable and untunable); default is the value it always had. | FLAT — 64/128/256 within noise on Zen3+Zen4+Zen5 (2026-08-21) |
| `trsm_r_fuse` | literal | Literal | side-R fuse threshold. NOW A KNOB (was a bare const, unpinnable and untunable); default is the value it always had. | TUNABLE and MIS-SPECIFIED — one scalar serves as BOTH the one-panel ceiling and the recursion leaf; no single value is right at n=128 and n=512 (Zen4 32 -> 0.900 vs 1.031). See task #169. |
| `trtri_base` | literal | Literal | triangular-inverse recursion base. NOW A KNOB (was a bare const, unpinnable and untunable); default is the value it always had. | FLAT — 8/16/32/64 within noise on all 3 uarchs, both trsm sides (2026-08-21) |
| `ztrsm_gt_base` | formula | Derived | formula over detected consts: L2 residency of A's KC×KC panel, `isqrt(_L2_BYTES ÷ sizeof(ComplexF64))`, min'd with the L1 stripe bound; fleet-validated on Zen4+Zen3 | — |
| `ztrsm_gt_mr` | formula | Derived | formula over detected consts: `_ZGT_W` | — |
| `ztrsm_zrt_nc` | formula | Derived | formula over detected consts: `prevpow(2, √(KC/2))` from the load-vs-fma balance at the base size, clamped by the register file `(nreg - 4) ÷ 2` | — |

## CPU detection

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `cpu_family` | other | Derived | the detected CPU display family itself (CPUID, via `cpumodel()`); the override exists for cross-compile and trim builds, not tuning. | n/a |
| `fast_cores` | other | Derived | the detected core count itself (`hw.perflevelN.physicalcpu`); the override exists for cross-compile and trim builds, not tuning. | n/a |
| `fp_datapath_bytes` | formula | Derived | the detected FP datapath width itself (CPUID Fn8000_001A FP128/FP256/FP512, else the register width); the preference exists for cross-compile and trim builds. | n/a |
| `l1d_assoc` | other | Derived | the detected L1d associativity itself (CPUID leaf 4 / 0x8000001D); the override exists for cross-compile and trim builds, not tuning. | n/a |
| `l3_bytes` | formula | Derived | the detected L3 size itself (`cachesize()`, floored at L2 where no L3/SLC is queryable, see above); the override exists for cross-compile and trim builds, not tuning. | n/a |
| `simd_bytes` | delegates | Derived | the detected SIMD width itself (`simdbytes()` on x86, the 16 B NEON floor elsewhere); the override exists for cross-compile and trim builds, not tuning. | n/a |
| `sme_f32` | other | Derived | the detected FEAT_SME2 + SME_F32F32 pair itself (`sysctlbyname`); the override exists for cross-compile and trim builds, not tuning. | n/a |
| `sme_f64` | other | Derived | the detected FEAT_SME2 + FEAT_SME_F64F64 pair itself (`sysctlbyname`); the override exists for cross-compile and trim builds, not tuning. | n/a |
| `sme_lanes` | formula | Derived | the detected streaming vector length itself (`hw.optional.arm.sme_max_svl_b`); the override exists for cross-compile and trim builds, not tuning. | n/a |
| `sme_lanes_f32` | formula | Derived | the detected streaming vector length itself (`hw.optional.arm.sme_max_svl_b`); the override exists for cross-compile and trim builds, not tuning. | n/a |

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
| `gbtrf_cmult` | literal | Measured | multiplier on the kl shape; the panel width itself is the shape, not the knob. | candidate |
| `gbtrf_cross` | delegates | Literal | crossover, derivation falsified; the C32 row is a 3-3 tie on Zen4. | candidate |
| `gbtrf_nb` | delegates | Derived | panel width from the kl shape; only the multiplier below it is measured. | n/a |

## LAPACK · lapack

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `chol_base_split` | formula | Derived | formula over detected consts: `_INTEL_AVX2` | — |
| `cpotf2_mr` | formula | Derived | formula over detected consts: `_at_cpotf2_mr(_HW)` | — |
| `cpotrf_base` | formula | Derived | 32 + 4*lanes: a width-independent overhead floor plus a per-lane slope. | n/a |
| `cpotrf_nbmax` | formula | Derived | formula over detected consts: `_at_cpotrf_nbmax(_HW)` | — |
| `potrf_base` | literal | Literal | recursion-overhead floor, µarch-flat at 16-32; the residency guess 64 is worse. | no |
| `potrf_base_f32` | sibling | Derived | sizeof ratio: F32 is half F64's bytes, so half the n. | n/a, follows potrf_base |
| `potrf_upper_direct_max` | delegates | Measured | a tiny-n crossover inside a noisy band; only host measurement resolves it. | candidate |

## LAPACK · lu

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `ciamax_simd_min` | formula | Literal | algorithm cost-ratio crossover (fixed SIMD setup vs per-element scalar scan); µarch- | — |
| `clu_flat_max` | literal | Literal | an ALGORITHM-SWITCH crossover (store-bound flat sweep vs blocked BLAS-3 update), with no | — |
| `key` | other | Literal | cost-ratio crossover between two ALGORITHMS (fixed SIMD setup vs per-element scalar | — |

## LAPACK · packed_chol

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `pptrf_blk_min` | literal | Literal | smallest n where unpacking pays; measured 2.26x at n=32, untested between 8 and 32. | candidate |
| `pptrf_blk_nb` | delegates | Derived | unset, the panel is the whole matrix (nb = n, table above); a pin caps the panel width and so the scratch. | n/a |
| `pptrf_spr_min` | delegates | Derived | unset, the cutoff is `2 * _vwidth(T)` per element type; a pin replaces it for every type. | n/a |

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
| `gemm_mt_work` | delegates | Measured | the join is an uncore latency and the flop rate needs a clock Julia does not report; both are measured on the host when the pool is created, and their product with `_MT_AMORTISE` is the floor. A pin replaces the measured floor. | calibrate_gemm_mt_work measures the fork-join round trip with its own pool and applies the same formula; DECLINES on a single-threaded process, and reports rather than pins a result outside ¼×…4× of the floor the library measured |
| `gemm_nc` | formula | Derived | formula over detected consts: `_at_gemm_nc(_HW)` | — |
| `gemm_nr` | formula | Derived | formula over detected consts: `_at_gemm_nr(_HW)` | — |
| `gemm_split_max` | formula | Derived | formula over detected consts: `_at_gemm_split_max(_HW)` | — |
| `gemm_unpack_max` | formula | Derived | formula over detected consts: `_at_gemm_unpack_max(_HW)` | — |
| `strassen` | formula | Derived | on exactly the detected vector widths (`_W64` 4 or 8) whose packing path implements Strassen; the flop cut itself is ISA-independent. | n/a |
| `strassen_base` | literal | Measured | the criterion is the column splits efficiency at a given blocks-per-worker, a scheduling-and-bandwidth property no detected const predicts, and the measured curve has no knee (5.3 blocks/worker 36%, 10.7 54%, 21 66%). `_gemm_mt_work()` gives only leaf >= 138, i.e. "is threading worth it at all", not "is it efficient". | none — orphaned; the threaded-harness blocker is GONE (calibrate_gemm_mt_work runs threaded), but this knob needs a split-EFFICIENCY criterion and the measured curve has no knee, so there is nothing yet for a calibrator to decide on; candidates _NR*2^j over 128..2048 |
| `strassen_maxdepth` | literal | Literal | accuracy budget (~3x error per level, type-independent); the performance side is | — |
| `strassen_min` | formula | Literal | split while min(m,n,k) >= this; the base stays >= ~min/2. | 512 GATE-REJECTED (Zen4-only, one cell, no miss to fix; table above) |
| `strassen_nopad` | literal | Literal | prefer depth-reduction over an O(n^2) pad; fleet table above, no-op off native AVX-512. | candidate |

## sme_kernel

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `sme_gemv_minwork` | formula | Derived | formula over detected consts: a quarter of L1 in elements, `_L1_BYTES ÷ (4 * sizeof(Float64))`, the panel size at which the per-block ZA fill and readback disappear into the stream. | — |
| `sme_gemvt_minm` | literal | Derived | the per-column ZA fill and readback is O(1) against O(m) of streamed column, so the crossover is a row count; it sits at the measured break against the NEON path it displaces. | sweep m at fixed n |
| `sme_gemvt_nc` | literal | Literal | a falsified-derivation literal: the criterion would be "widest NC that still saves x traffic", and it predicts 8, which measures WORSE at every size. Four is what the table above says. | candidate, (4,8) |
| `sme_gemvt_resident_max` | formula | Derived | L2 residency of A, m*n*8 <= 3/4 * _L2_BYTES; the quarter left over is x (re-read per column block), y, the deferred strips and the core's own lines; measured crossover 15 MB of 20. | candidate, sweep m*n at fixed m across _L2_BYTES |
| `sme_ger_groups` | literal | Measured | each group carries its own load, accumulates and store, so this is stream count and dependency depth together, not residency or width. | none — orphaned; candidates (1,2,4,8) |
| `sme_ger_minm` | literal | Derived | formula over detected consts: the kernel's own row granularity, `4 * _SME_L`, below and outside of which it has nothing to run. | — |
| `sme_inplace_max` | formula | Derived | A-block residency against the detected L1: an in-place walk stays as cheap as a contiguous stream while the block the kernel re-reads is a couple of L1-fuls, and the coefficient is set one step inside where that was measured to flip under a worst-case stride, because a caller with extra cache traffic tips the edge. | n/a, follows _L1_BYTES |
| `sme_inplace_njp` | literal | Measured | the column-panel re-read count at which packing A starts to pay, for a stride whose columns spread over two or more L1 sets; neither block size nor page geometry places it, and the value is bounded above by the gate cells that lose when the cut admits their own panel count. `_sme_inplace_njp` halves it for a one-set stride, which it identifies from the detected `_L1_WAY_BYTES`. | 2 s; _EXPINT[10] sweeps it by hand |
| `sme_min` | literal | Measured | tile-occupancy crossover, not a residency formula: the general cut is where the worst remainder (1) starts paying, and exact multiples of MR are admitted earlier because they pack no remainder panel at all. | none — orphaned; sweep |
| `sme_min_exact` | literal | Measured | the same occupancy crossover for shapes that pack no remainder panel at all; one full row panel already pays, per the table above. | none — orphaned; sweep |
| `sme_min_exact32` | literal | Measured | the occupancy crossover for ragged Float32 shapes, selected from the candidate set "multiples of `_SME_MR32`" by sweeping DEPTH as well as extent: at k = 64 and 256 the 1.25-tile and 1.5-tile shapes win 1.12-1.74, and at k = 10 they lose 0.34 and 0.48, so the multiplier that agrees with the Float64 one is a result here rather than an inheritance. | none — orphaned; sweep |
| `sme_panel_bytes` | literal | Measured | a packing-memory ceiling, not a residency criterion: KC is grown to minimize C passes, and where a larger panel stops paying depends on packing throughput against kernel throughput. | none — orphaned; sweep |

## sme_l1

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `sme_asum_min` | literal | Measured | this operation's own cold crossing, 131072 against dot's 65536; the four level-1 floors do not share one and no stream count predicts the ordering. | 4.2 s |
| `sme_axpy_min` | literal | Measured | this operation's own cold crossing, 131072, half of scal's; the four level-1 floors do not share one and no stream count predicts the ordering. | 4.2 s |
| `sme_dot_min` | literal | Measured | where a fixed ZA prologue disappears into the stream; a ratio between two kernels, not a residency criterion. | 4.2 s |
| `sme_l1_groups` | literal | Measured | every fmla into one ZA slice is a serial chain, so this is dependency depth, not residency or width; the optimum inverts either side of two. | none — orphaned; candidates (1,2,4,8) |
| `sme_rmw_groups` | literal | Measured | each group carries its own load and store, so this is stream count and dependency depth together, not residency or width. | none — orphaned; candidates (1,2,4,8) |
| `sme_scal_min` | literal | Measured | where a fixed ZA prologue disappears into the stream on a single cold pass; a ratio between two kernels, not a residency criterion. | 4.2 s |

## workspace

| Knob | Default | Tier | Why | `tune!()` |
|---|---|---|---|---|
| `l3_nb` | formula | Derived | formula over detected consts: `clamp(_round_dn(isqrt(_L2_BYTES ÷ 32), 16), 16, 128` | — |

---

Const names, defaults and files are deliberately NOT tabulated: they are one `grep` away
and made this table too wide to read. The knob key is the identifier that matters.

## Orphaned Measured knobs

9 of the 24 Measured knobs have no entry in
`bench/calibrate.jl`, so `tune!()` cannot set them however many times it runs. The tier is
still correct — the value was reached by measurement — but the pin it asks for is
unreachable until a calibrator exists. Each row's `tune!()` cell reads `none`, and the
handle after it is what a person would sweep by hand meanwhile.

| Knob | Hand-sweep handle |
|---|---|
| `symv_sme_min` | sweep n |
| `strassen_base` | the threaded-harness blocker is GONE (calibrate_gemm_mt_work runs threaded), but this knob needs a split-EFFICIENCY criterion and the measured curve has no knee, so there is nothing yet for a calibrator to decide on; candidates _NR*2^j over 128..2048 |
| `sme_ger_groups` | candidates (1,2,4,8) |
| `sme_min` | sweep |
| `sme_min_exact` | sweep |
| `sme_min_exact32` | sweep |
| `sme_panel_bytes` | sweep |
| `sme_l1_groups` | candidates (1,2,4,8) |
| `sme_rmw_groups` | candidates (1,2,4,8) |

## Tuning constants that are NOT knobs

34 `const _X = <literal>` values in `src/` with no `@load_preference`.
They are tuning constants all the same — and in a WORSE position than a knob, because
they cannot be pinned, cannot be tuned by `tune!()`, and were invisible to the audit
above. `trtrs` is the worked example: its real path (trsm side-L) runs almost entirely
on these, not on knobs.

**Tier:** 31 Literal · 3 Unaudited.


### BLAS-1 SIMD kernels

| Const | Value | Tier | Why |
|---|---|---|---|
| `_IAMAX_NB_STREAM` | 4 | Literal | DERIVABLE, not yet derived: ILP chain count, ISA-invariant by the latency x throughput argument. |
| `_UNROLL` | 4 | Literal | DERIVABLE, not yet derived: 4 chains x W lanes, an ILP count tied to _ILP_TARGET. |

### BLAS-2 (gemv/ger/trmv/trsv)

| Const | Value | Tier | Why |
|---|---|---|---|
| `_CGEMV_NP` | 8 | Literal | DERIVABLE, not yet derived: complex gemv-N panel width, same register argument as _GEMV_NP. |
| `_GEMVN_MINNER_U` | 4 | Literal | row-unroll paired with the gemvn_minner knob; moves with it, not independently. |
| `_GEMVT_NC_DEEP` | 8 | Literal | DERIVABLE, not yet derived: one x-load per 8 FMAs, the load:FMA ratio clearing the MLP plateau; _NVREG-guarded. |
| `_GEMVT_U_DEEP` | 4 | Literal | DERIVABLE, not yet derived: 4 lines per stream, so NC*U = 32 lines named — enough to cover L2. |
| `_GEMV_NP` | 8 | Literal | DERIVABLE, not yet derived: gemv-N panel width; the comment already reasons in MR and register pressure. |
| `_GER_PANEL_U` | 4 | Literal | its own comment calls it 'a genuine tuning knob'. TUNABLE, and never made one. |
| `_SYMV_MR` | 2 | Unaudited | — |
| `_SYMV_NB` | 8 | Literal | FLEET-VALIDATED 2026-08-21: a CAP, and the consumer's `min(_SYMV_NB, _vwidth(T))` is what makes it right (8 on AVX-512, 4 on AVX2). Halving NB costs 10-27% on all 3 boxes, so the cap binds and the value is not arbitrary. |
| `_SYMV_NBLK` | 8 | Literal | one value suits every µarch: the shape bound 2/(nblk+1) is arithmetic, and the fold tax table above is flat across nblk on all three boxes. |
| `_TRSV_T_F` | 8 | Literal | trsv-T fuse factor; the routing bound is expressed as a multiple of it. TUNABLE. |

### CPU detection

| Const | Value | Tier | Why |
|---|---|---|---|
| `_GEMM_SPLIT_S` | 2 | Literal | split-reduction factor, measured crossover ~n=56 on Zen4. TUNABLE. |
| `_ILP_TARGET` | 16 | Literal | 2 x FMA latency(4) x FMA ports(2); an architectural assumption, and neither latency nor port count is detectable. |

### LAPACK · eigen_dc

| Const | Value | Tier | Why |
|---|---|---|---|
| `_STEDC_NB` | 25 | Literal | LAPACK's SMLSIZ: algorithm-intrinsic, machine-independent. tuning.md §4. |

### LAPACK · hessenberg

| Const | Value | Tier | Why |
|---|---|---|---|
| `_GEHRD_UNBLK_MAX` | 128 | Literal | LAPACK ILAENV nx for dgehrd, carried over UNMEASURED here; owed a fleet crossover sweep. |

### LAPACK · lapack

| Const | Value | Tier | Why |
|---|---|---|---|
| `_CHOL_TILE` | 4 | Literal | the column count the fused Cholesky base kernels are hand-unrolled for; a structural invariant of that code. |
| `_TR_TB` | 32 | Literal | residency-INVARIANT: two 32^2 F64 tiles = 16 KB, under any real L1. Deriving it would change nothing. |

### LAPACK · laqr

| Const | Value | Tier | Why |
|---|---|---|---|
| `_LAQR_KEXNW` | 5 | Literal | LAPACK dlaqr0 KEXNW, algorithm-intrinsic and machine-independent. |
| `_LAQR_KEXSH` | 6 | Literal | LAPACK dlaqr0 KEXSH, algorithm-intrinsic and machine-independent. |
| `_LAQR_KNWSWP` | 500 | Literal | LAPACK ispec 13 window-widening threshold, algorithm-intrinsic and machine-independent. |
| `_LAQR_NIBBLE` | 14 | Literal | LAPACK ispec 14 (NIBBLE), algorithm-intrinsic and machine-independent. |
| `_LAQR_NMIN` | 200 | Literal | dlahqr-vs-dlaqr0 crossover; derivation falsified, fleet table above is the evidence. |
| `_LAQR_SKIPMIN` | 75 | Literal | LAPACK dlaqr0/ILAENV convergence constant, algorithm-intrinsic and machine-independent. |

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

### LAPACK · svd_dc

| Const | Value | Tier | Why |
|---|---|---|---|
| `_DC_THRESHOLD` | 64 | Literal | divide-and-conquer cut. TUNABLE, algorithm-intrinsic so no formula. |
| `_SEC_BISECT_CAP` | 0 | Literal | secular-equation bisection cap; 0 disables. TUNABLE. |

### gemm (BLAS-3)

| Const | Value | Tier | Why |
|---|---|---|---|
| `_GEMM_TINY` | 6 | Literal | below this the naive loop beats the packed path. TUNABLE. |
| `_MT_AMORTISE` | 8 | Unaudited | — |

### sme_kernel

| Const | Value | Tier | Why |
|---|---|---|---|
| `_SME_GEMVT_ZA_NC` | 8 | Unaudited | — |
