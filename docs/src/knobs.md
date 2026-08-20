# The Knob Registry

!!! warning "Generated file — do not edit"
    Produced by `test/knob_registry.jl` from `src/`. To change a row, change the code
    or its `# PDM:` marker and regenerate. `test/knob_registry_tests.jl` fails if this
    file is out of date.

Every `@load_preference` key in `src/` — 118 of them. The PDM ladder
(`docs/src/tuning.md`) requires each to be **Derived** (default is a formula over detected
consts) or **Measured** (it is not — which needs a justification and a `tune!()` cost).

**Syntactic tier** is what the generator can see in the default expression; it is a
*classification aid, not a verdict*. `Literal` means "a bare number is the default" — it
may be a proven invariant (fine), a falsified derivation (fine, documented), or unconverted
debt. The `# PDM:` marker is the human judgement and is the column that matters.

| Syntactic tier | Count |
|---|---|
| Coupled | 6 |
| Derived | 42 |
| Flag | 4 |
| Literal | 34 |
| Other | 1 |
| Predicate-keyed | 11 |
| Pref-gated | 20 |

**Audited: 4 / 118** knobs carry a `# PDM:` marker. The rest are the
worklist — an unaudited knob is one nobody has justified, which is exactly how redundant
and duplicated knobs survive.


## BLAS-1 SIMD kernels

| Key | Const | Default | Tier | PDM justification |
|---|---|---|---|---|
| `axpy_dram` | `_AXPY_DRAM` | `_at_axpy_dram(_HW))::Int` | Derived | Derived — arm 208 is the narrow 256-bit phase kernel, correct exactly where 512-bit ops are double-pumped over a 256-bit datapath. Mechanism named, not a fleet fit: +17% on Zen4, LOSES on Zen3, and Zen5 (native 512, not double-pumped) measures 4. Falsifier: a non-double-pumped box preferring 208. \| tune: n/a — Derived |
| `axpy_unroll` | `_AXPY_BAND` | `_at_axpy_band(_HW))::Int` | Derived | *unaudited* |
| `zaxpy_narrow` | `_ZAXPY_NARROW_PREF` | `nothing)` | Pref-gated | *unaudited* |

## BLAS-2 (gemv/ger/trmv/trsv)

| Key | Const | Default | Tier | PDM justification |
|---|---|---|---|---|
| `cgemv_mr` | `_CGEMV_MR` | `4)::Int` | Literal | *unaudited* |
| `cgemv_rb` | `_CGEMV_RB` | `_L2_BYTES ÷ 16)::Int   # m·n complex threshold for one-panel mode` | Derived | *unaudited* |
| `cgemvn_nc` | `_CGEMVN_NC` | `4)::Int             # columns per panel (OB uses 4)` | Literal | *unaudited* |
| `cgemvn_nc_big` | `_CGEMVN_NC_BIG` | `3 * _vwidth(Float64) ÷ 2)::Int` | Derived | *unaudited* |
| `cgemvn_pf` | `_CGEMVN_PF` | `_vwidth(Float64) == 4)::Bool  # A-stream prefetch (AVX2)` | Derived | *unaudited* |
| `cgemvt_cfg` | `_CGEMVT_CFG` | `_CGEMVT_CFG_BIG)::Int` | Coupled | *unaudited* |
| `cgemvt_half` | `_CGEMVT_HALF` | `_vwidth(Float64) == 4)::Bool` | Derived | *unaudited* |
| `cgemvt_nc` | `_CGEMVT_NC` | `4)::Int   # legacy pin; superseded by _cgemvt_cfg()` | Literal | *unaudited* |
| `cgemvt_pf` | `_CGEMVT_PF` | `_vwidth(Float64) == 4)::Bool` | Derived | *unaudited* |
| `gemvn_mb` | `_GEMVN_MB` | `max(_vwidth(Float64), _L1_BYTES ÷ 2 ÷ sizeof(Float64)))::Int  # m-block: y-block ≤ ½L1 stays resident while sweeping all n columns` | Derived | *unaudited* |
| `gemvn_minner` | `_GEMVN_MINNER_PREF` | `nothing)` | Pref-gated | *unaudited* |
| `gemvn_minner_maxa` | `_GEMVN_MINNER_MAXA` | `4 * _L3_BYTES)::Int  # max A bytes (m·n·sizeof) for minner` | Derived | *unaudited* |
| `gemvn_np_narrow` | `_GEMVN_NP_NARROW` | `max(2, _L1D_ASSOC - 2))::Int` | Derived | *unaudited* |
| `gemvn_rb` | `_GEMVN_RB` | `_vwidth(Float64) == 4 ? 64 : 448)::Int  # gemv-N: n ≤ this → row-block; larger → column-panel. AVX2 cut dropped 192→64: with _GEMV_MR=8 the sequential-streaming panel path now beats strided row-block for all n≥96 (128: 0.92→1.0); row-block only wins at n≤64 where panel's m<mr all-masked tail dominates. Zen4 1MB L2 → 448.` | Derived | *unaudited* |
| `gemvt_deep` | `_GEMVT_DEEP_PREF` | `nothing)` | Pref-gated | *unaudited* |
| `gemvt_nc` | `_GEMVT_NC_PREF` | `nothing)` | Pref-gated | Derived-falsified — the project's own latency x throughput criterion (_ILP_TARGET) predicts EIGHT chains; every fleet box measures 4 best, monotonically worse at 8 and 16. A derivation that predicts the wrong answer is falsified, so the literal 4 ships with its table. Re-open only via the NC x U pair (AOCL runs NC=8 x U=4), which the 2026-08-20 grid found ties in the DRAM regime. \| tune: not a tune!() candidate — measured µarch-INVARIANT (4 on all three boxes) |
| `gemvt_perscan` | `_GEMVT_PERSCAN_PREF` | `nothing);` | Pref-gated | Measured — no detected const partitions the fleet: EVERY PAIR of boxes disagrees at some size. L1 was the last candidate and is FALSIFIED — wintermute and galen both have 32 KiB L1 and want opposite arms at n=512 (percol 1.113 vs blocked 0.953/0.964/0.973, three processes each). Not L2, not L3, not SIMD width either (wintermute/neuromancer share width 64 and disagree at n=1024). A `_wide_simd` derivation was written and falsified before shipping. \| tune: not implemented; the knob is also too COARSE — the optimum is per-(box,size), so galen leaves ~1.9% at n=1024 and Zen5 ~3.9% at n=512 (#160). Needs a per-SIZE route pin, a knob-shape change, not a value. |
| `gemvt_pf` | `_GEMVT_PF_PREF` | `nothing)` | Pref-gated | *unaudited* |
| `gemvt_u` | `_GEMVT_U_PREF` | `nothing)` | Pref-gated | *unaudited* |
| `ger_panel_np` | `_GER_NP` | `nothing), 1)::Int   # req8-ok: minimax, table above` | Pref-gated | Measured — optimum is 8/4/1 on Zen4/Zen3/Zen5; Zen4 and Zen3 agree on L2, L3, SIMD width and register count, so no formula over detected consts separates them. Depends on DRAM write-stream count, which is not a detected const. \| tune: ~22 s measured (4 candidate arms x 8 rounds at n=2048, DRAM regime) |
| `tri_c_blk_min` | `_TRI_C_BLK_MIN` | `_vwidth(Float64) == 4 ? 256 : 1024)::Int` | Derived | *unaudited* |
| `tri_c_t_unb` | `_TRI_C_T_UNB` | `1024)::Int` | Literal | *unaudited* |
| `tri_nb` | `_TRI_NB` | `clamp(_round_dn(isqrt(_L1_BYTES ÷ 8), 16), 16, 64))::Int` | Derived | *unaudited* |
| `tri_t_unb` | `_TRI_T_UNB` | `512)::Int` | Literal | *unaudited* |
| `trmv_f_dram` | `_TRMV_F_DRAM` | `4)::Int          # req8-ok: Measure-tier debt, see above` | Literal | *unaudited* |
| `trmv_f_switch` | `_TRMV_F_SWITCH` | `2)::Int      # req8-ok: Measure-tier debt, see above` | Literal | *unaudited* |
| `trmv_fused_min` | `_TRMV_FUSED_MIN_PREF` | `nothing)` | Pref-gated | *unaudited* |
| `trsv_reg_max` | `_TRSV_REG_MAX` | `_SCALAR_FPREGS - 4)::Int` | Derived | *unaudited* |
| `zhemv_pf` | `_ZHEMV_PF` | `_vwidth(Float64) == 4)::Bool` | Derived | *unaudited* |
| `zhemv_pf_tiles` | `_ZHEMV_PF_TILES` | `8)::Int` | Literal | *unaudited* |

## BLAS-2 banded

| Key | Const | Default | Tier | PDM justification |
|---|---|---|---|---|
| `gbmv_conv_max` | `_GBMV_CONV_MAX` | `_vwidth(Float64) == 4 ? 20 : 48)::Int` | Derived | *unaudited* |

## BLAS-2 packed

| Key | Const | Default | Tier | PDM justification |
|---|---|---|---|---|
| `spmv_panel` | `_SPMV_PANEL` | `true)::Bool   # ON: locked-fleet A/B win — flattens the decline on all 3 µarchs` | Flag | *unaudited* |
| `spmv_panel_minap` | `_SPMV_PANEL_MINAP` | `_L1_BYTES)::Int  # min AP bytes (n(n+1)/2·sizeof) to panel — fires ≈ n≥128 on 32KB L1` | Derived | *unaudited* |

## BLAS-3 (trmm/trsm/syrk/symm)

| Key | Const | Default | Tier | PDM justification |
|---|---|---|---|---|
| `chemm_pack_cut` | `_CHEMM_PACK_CUT` | `_vwidth(Float64) == 4 ? 4096 : 32)::Int` | Derived | *unaudited* |
| `csymm_pack_cut` | `_CSYMM_PACK_CUT` | `_CHEMM_PACK_CUT)::Int` | Coupled | *unaudited* |
| `csyr2k_fused_max` | `_CSYR2K_FUSED_MAX` | `_vwidth(Float64) == 4 ? 192 : 0)::Int` | Derived | *unaudited* |
| `csyr2k_pack_cut` | `_CSYR2K_PACK_CUT` | `_vwidth(Float64) == 4 ? 8 : 8)::Int` | Derived | *unaudited* |
| `csyrk_3m_min` | `_CSYRK_3M_MIN` | `128)::Int` | Literal | *unaudited* |
| `csyrk_pack_cut` | `_CSYRK_PACK_CUT` | `16)::Int        # trans='N': recursion below this` | Literal | *unaudited* |
| `csyrk_pack_cut_t` | `_CSYRK_PACK_CUT_T` | `4)::Int     # trans='C'/'T': packed ~always` | Literal | *unaudited* |
| `csyrk_unified_max` | `_CSYRK_UNIFIED_MAX` | `_vwidth(Float64) == 4 ? 512 : 0)::Int` | Derived | *unaudited* |
| `csyrk_unpack_max` | `_CSYRK_UNPACK_MAX` | `_vwidth(Float64) == 4 ? 16 : 192)::Int` | Derived | *unaudited* |
| `ctrmm_pack` | `_CTRMM_PACK` | `_vwidth(Float64) == 4)::Bool` | Derived | *unaudited* |
| `ctrmm_pack_min` | `_CTRMM_PACK_MIN` | `48)::Int` | Literal | *unaudited* |
| `ctrsm_direct_max` | `_CTRSM_DIRECT_MAX` | `64)::Int` | Literal | *unaudited* |
| `ctrsm_ncut` | `_CTRSM_NCUT` | `128)::Int   # B-width cut: ≤ → narrow (j-outer recursion)` | Literal | *unaudited* |
| `ctrsm_rec_l` | `_CTRSM_REC_L` | `64)::Int` | Literal | *unaudited* |
| `gemmtrsm_mr` | `_GT_MR` | `min(8, (_GT_NREG - _GT_NRV - 2) ÷ _GT_NRV))::Int` | Predicate-keyed | *unaudited* |
| `gemmtrsm_nrv` | `_GT_NRV` | `_GT_NREG >= 32 ? 3 : 2)::Int` | Predicate-keyed | *unaudited* |
| `symm_pack_cut` | `_SYMM_PACK_CUT` | `_at_symm_mat_max(_HW))::Int` | Derived | *unaudited* |
| `syr2k_2pass` | `_SYR2K_2PASS` | `_vwidth(Float64) == 4 ? 128 : typemax(Int))::Int` | Derived | *unaudited* |
| `syr2k_mr` | `_SYR2K_MR` | `_vwidth(Float64) == 4 ? 2 : _MR)::Int` | Derived | *unaudited* |
| `syr2k_nr` | `_SYR2K_NR` | `_NR)::Int` | Coupled | *unaudited* |
| `syr2k_pack_cut` | `_SYR2K_PACK_CUT` | `_at_rank_k_pack_cut(_HW))::Int` | Derived | *unaudited* |
| `syrk_dbase` | `_SYRK_DBASE` | `32)::Int` | Literal | *unaudited* |
| `syrk_mr` | `_SYRK_MR` | `2)::Int` | Literal | *unaudited* |
| `syrk_pack_cut` | `_SYRK_PACK_CUT` | `_at_rank_k_pack_cut(_HW))::Int` | Derived | *unaudited* |
| `syrk_unified_max` | `_SYRK_UNIFIED_MAX` | `_vwidth(Float64) == 4 ? 48 : 0)::Int` | Derived | *unaudited* |
| `trmm_ddirect` | `_TRMM_DDIRECT` | `4)` | Literal | *unaudited* |
| `trmm_pack_min` | `_TRMM_PACK_MIN` | `(5 * _GEMM_UNPACK_MAX) ÷ 2)::Int` | Other | *unaudited* |
| `trmm_rkc` | `_TRMM_RKC` | `_KC)::Int` | Coupled | *unaudited* |
| `trmm_rpack` | `_TRMM_RPACK` | `448)::Int` | Literal | *unaudited* |
| `trsm_narrow_max` | `_TRSM_NARROW_MAX` | `4)::Int` | Literal | *unaudited* |
| `ztrsm_gt_mr` | `_ZGT_MR` | `_ZGT_W)::Int` | Predicate-keyed | *unaudited* |

## CPU detection

| Key | Const | Default | Tier | PDM justification |
|---|---|---|---|---|
| `force_hooks` | `_FORCE_HOOKS` | `true)::Bool` | Flag | *unaudited* |
| `simd_bytes` | `_SIMD_BYTES` | `nothing)` | Pref-gated | *unaudited* |

## LAPACK · banded_chol

| Key | Const | Default | Tier | PDM justification |
|---|---|---|---|---|
| `pbtrf_cross_kd` | `_PBTRF_CROSS_PREF` | `nothing)` | Pref-gated | *unaudited* |
| `pbtrf_nb` | `_PBTRF_NB_PREF` | `nothing)` | Pref-gated | *unaudited* |
| `pbtrf_nb_small` | `_PBTRF_NBS_PREF` | `nothing)` | Pref-gated | *unaudited* |
| `pbtrf_u_native_kd` | `_PBTRF_UCROSS_PREF` | `nothing)` | Pref-gated | *unaudited* |

## LAPACK · bunchkaufman

| Key | Const | Default | Tier | PDM justification |
|---|---|---|---|---|
| `sytrf_cmult` | `_SYTRF_CMULT_PREF` | `nothing)` | Pref-gated | *unaudited* |
| `sytrf_nb` | `_SYTRF_NB_PREF` | `nothing)   # pin-ok: user override, no benchmark behind it` | Pref-gated | *unaudited* |

## LAPACK · gbtrf

| Key | Const | Default | Tier | PDM justification |
|---|---|---|---|---|
| `gbtrf_cross` | `_GBTRF_CROSS_PREF` | `nothing)` | Pref-gated | *unaudited* |
| `gbtrf_nb` | `_GBTRF_NB_PREF` | `nothing)` | Pref-gated | *unaudited* |

## LAPACK · lapack

| Key | Const | Default | Tier | PDM justification |
|---|---|---|---|---|
| `chol_base_split` | `_CHOL_BASE_SPLIT` | `_INTEL_AVX2)::Bool` | Predicate-keyed | *unaudited* |
| `chol_nb` | `_CHOL_NB` | `4)::Int   # trsm panel column block (register-tile width)` | Literal | *unaudited* |
| `chol_nc` | `_CHOL_NC` | `4)::Int   # syrk column block (register-tile width)` | Literal | *unaudited* |
| `cpotf2_mr` | `_CPOTF2_MR` | `_at_cpotf2_mr(_HW))::Int   # req#8: derived 64÷datapath_bytes (2 double-pump/AVX2, 1 native-512)` | Derived | *unaudited* |
| `cpotrf_base` | `_CPOTRF_BASE` | `_at_cpotrf_base(_HW))::Int   # req#8: derived 32+4·W (48/64)` | Derived | *unaudited* |
| `cpotrf_nbmax` | `_CPOTRF_NBMAX` | `_at_cpotrf_nbmax(_HW))::Int   # req#8: derived 64+16·W (128/192)` | Derived | *unaudited* |
| `potrf_base` | `_POTRF_BASE` | `32)::Int` | Literal | *unaudited* |
| `potrf_base_f32` | `_POTRF_BASE_F32` | `_POTRF_BASE >> 1)::Int` | Coupled | *unaudited* |
| `potrf_pad` | `_POTRF_PAD` | `true)::Bool   # disable to A/B the pad's benefit per µarch` | Flag | *unaudited* |
| `potrf_upper_direct_max` | `_POTRF_UDIRECT_PREF` | `nothing)` | Pref-gated | *unaudited* |

## LAPACK · packed_chol

| Key | Const | Default | Tier | PDM justification |
|---|---|---|---|---|
| `pptrf_blk_min` | `_PPTRF_BLK_MIN` | `16)::Int` | Literal | *unaudited* |
| `pptrf_blk_nb` | `_PPTRF_BLK_NB` | `_LU_NB)::Int` | Coupled | *unaudited* |
| `pptrf_spr_min` | `_PPTRF_SPR_PREF` | `0)::Int   # req8-ok: unset sentinel, not a tuning value` | Literal | *unaudited* |

## LAPACK · pstrf

| Key | Const | Default | Tier | PDM justification |
|---|---|---|---|---|
| `pstrf_fuse_max` | `_PSTRF_FUSE_MAXL` | `(_L1_BYTES ÷ 2) ÷ _CACHELINE)::Int` | Derived | *unaudited* |
| `pstrf_rowcache_min` | `_PSTRF_ROWCACHE_PREF` | `128)` | Literal | *unaudited* |

## LAPACK · qr

| Key | Const | Default | Tier | PDM justification |
|---|---|---|---|---|
| `qr_nb_c` | `_QR_NB_C` | `32)::Int` | Literal | *unaudited* |

## LAPACK · svd

| Key | Const | Default | Tier | PDM justification |
|---|---|---|---|---|
| `brd_nb` | `_BRD_NB_PREF` | `nothing)` | Pref-gated | *unaudited* |
| `bt_nb` | `_BT_NB` | `32)::Int` | Literal | *unaudited* |

## gemm (BLAS-3)

| Key | Const | Default | Tier | PDM justification |
|---|---|---|---|---|
| `cgemm_3m` | `_CGEMM_3M` | `true)::Bool` | Flag | *unaudited* |
| `cgemm_3m_kmin` | `_CGEMM_3M_KMIN` | `16)::Int  # min(m,n,k) ≥ this (thin gemms: overhead dominates)` | Literal | *unaudited* |
| `cgemm_3m_max` | `_CGEMM_3M_MAX` | `2048)::Int  # max(m,n,k) ≤ this` | Literal | *unaudited* |
| `cgemm_3m_min` | `_CGEMM_3M_MIN` | `48)::Int    # max(m,n,k) ≥ this` | Literal | *unaudited* |
| `cgemm_kc` | `_CKC` | `_l1_block(_HW, ComplexF64, max(_CNR, _CNR_SMALL)))::Int` | Derived | *unaudited* |
| `cgemm_mr` | `_CMR` | `_W64 == 4 ? 1 : 2)::Int` | Predicate-keyed | *unaudited* |
| `cgemm_nr` | `_CNR` | `_W64 == 4 ? 6 : 4)::Int` | Predicate-keyed | *unaudited* |
| `cgemm_nr_small` | `_CNR_SMALL` | `_W64 == 4 ? 4 : _CNR)::Int` | Predicate-keyed | *unaudited* |
| `cgemm_nrsmall_max` | `_CGEMM_NRSMALL_MAX` | `_W64 == 4 ? 64 : 0)::Int` | Predicate-keyed | *unaudited* |
| `cgemm_tiny` | `_CGEMM_TINY` | `6)::Int` | Literal | *unaudited* |
| `cgemm_unpack_max` | `_CGEMM_UNPACK_MAX` | `_W64 == 4 ? 40 : 192)::Int` | Predicate-keyed | *unaudited* |
| `cuker_nr6_min` | `_CUKER_NR6_MIN` | `_W64 == 4 ? 48 : typemax(Int))::Int` | Predicate-keyed | *unaudited* |
| `gemm_kc` | `_KC` | `_at_gemm_kc(_HW))::Int   # B micropanel kc·_NR·8 ≤ ½·L1 (BLIS)` | Derived | *unaudited* |
| `gemm_mr` | `_MR` | `_at_gemm_mr(_HW))::Int` | Derived | *unaudited* |
| `gemm_mr1_max` | `_GEMM_MR1_MAX` | `_at_gemm_mr1_max(_HW))::Int` | Derived | *unaudited* |
| `gemm_nc` | `_NC` | `_at_gemm_nc(_HW))::Int   # B col block ≤ ¼·L3, po2-dodged` | Derived | *unaudited* |
| `gemm_nr` | `_NR` | `_at_gemm_nr(_HW))::Int` | Derived | *unaudited* |
| `gemm_split_max` | `_GEMM_SPLIT_MAX` | `_at_gemm_split_max(_HW))::Int` | Derived | *unaudited* |
| `gemm_unpack_max` | `_GEMM_UNPACK_MAX` | `_at_gemm_unpack_max(_HW))::Int` | Derived | *unaudited* |
| `strassen` | `_STRASSEN` | `_W64 == 4 \|\| _W64 == 8)::Bool` | Predicate-keyed | *unaudited* |
| `strassen_maxdepth` | `_STRASSEN_MAXDEPTH` | `3)::Int` | Literal | *unaudited* |
| `strassen_min` | `_STRASSEN_MIN` | `1024)::Int      # split while min(m,n,k) ≥ this` | Literal | *unaudited* |

## workspace

| Key | Const | Default | Tier | PDM justification |
|---|---|---|---|---|
| `l3_nb` | `_L3_NB` | `clamp(_round_dn(isqrt(_L2_BYTES ÷ 32), 16), 16, 128))::Int` | Derived | *unaudited* |
