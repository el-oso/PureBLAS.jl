# LAPACK / BLAS coverage

Which `LinearAlgebra` operations route to PureBLAS after `PureBLAS.activate()`, and how they measure.

Every number is a **PB / max(OpenBLAS, AOCL) speed ratio**: the routine's worst (op, size) cell on that
box, one column per microarchitecture. `≥ 1.0` **to two significant digits** gates (so `0.995` passes,
`0.9949` does not — see [Methodology](methodology.md)); a miss is printed at three digits, floored, and
bolded (routing tables) or colour-banded (BLAS tables). **Routes** = forwards to PureBLAS via LBT after
`activate()`; ⏳ = not gated yet. Types: **s** = Float32, **d** = Float64, **c** = ComplexF32,
**z** = ComplexF64.

Measurement, provenance and the caveats that change how a cell should be read:
[Methodology](methodology.md). Per-routine analysis and history: [Notes](notes.md).

## BLAS

One row per routine, generated from the caches by `bench/coverage_ops.jl`; `n=` names the worst cell.

```@raw html
<style>
.pbg{--l:#e3e6ee;--m:#5d6675;--ok:#1f8a5b;--b1:#7a8496;--b2:#c07d12;--b3:#cf5a35;--b4:#b3243a;
 --okbg:#e9f6ef;--b1bg:#f1f3f7;--b2bg:#fdf3e2;--b3bg:#fceee9;--b4bg:#fbe9ed;
 border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;display:table}
html.dark .pbg{--l:#242c3b;--m:#98a1b3;--ok:#4cc98d;--b1:#8891a3;--b2:#e0a63c;--b3:#f08055;--b4:#ff5f7a;
 --okbg:#12271d;--b1bg:#1a2130;--b2bg:#2a2113;--b3bg:#2c1a15;--b4bg:#2c1420}
@media (prefers-color-scheme:dark){html:not(.light) .pbg{--l:#242c3b;--m:#98a1b3;--ok:#4cc98d;--b1:#8891a3;
 --b2:#e0a63c;--b3:#f08055;--b4:#ff5f7a;--okbg:#12271d;--b1bg:#1a2130;--b2bg:#2a2113;--b3bg:#2c1a15;--b4bg:#2c1420}}
.pbg th,.pbg td{border-bottom:1px solid var(--l);padding:7px 12px;text-align:left}
.pbg thead th{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--m);white-space:nowrap}
.pbg tbody th{font-weight:400}
.pbg td{border-left:1px solid var(--l);white-space:nowrap}
.pbg .v{font-weight:600}
.pbg .n{font-size:.82em;color:var(--m);margin-left:7px}
.pbg td.ok{background:var(--okbg)} .pbg td.ok .v{color:var(--ok)}
.pbg td.b1{background:var(--b1bg)} .pbg td.b1 .v{color:var(--b1)}
.pbg td.b2{background:var(--b2bg)} .pbg td.b2 .v{color:var(--b2)}
.pbg td.b3{background:var(--b3bg)} .pbg td.b3 .v{color:var(--b3)}
.pbg td.b4{background:var(--b4bg)} .pbg td.b4 .v{color:var(--b4)}
.pbg-key{display:flex;flex-wrap:wrap;gap:14px;margin:10px 0 0;font-size:12px;color:#5d6675}
html.dark .pbg-key{color:#98a1b3}
.pbg-key i{font-style:normal;display:inline-block;width:11px;height:11px;border-radius:2px;
 margin-right:5px;vertical-align:-1px}
</style>
```


#### BLAS-1

```@raw html
<table class="pbg"><thead><tr><th>routine</th><th>Zen3 · AVX2</th><th>Zen4 · AVX-512</th><th>Zen5 · AVX-512</th></tr></thead><tbody>
<tr><th><code>asum</code></th><td class="ok"><span class="v">1.34</span></td><td class="b2"><span class="v">0.988</span><span class="n">n=1000000</span></td><td class="b2"><span class="v">0.959</span><span class="n">n=1000000</span></td></tr>
<tr><th><code>axpy</code></th><td class="ok"><span class="v">1.01</span></td><td class="ok"><span class="v">1.0</span></td><td class="b2"><span class="v">0.959</span><span class="n">n=300000</span></td></tr>
<tr><th><code>dot</code></th><td class="ok"><span class="v">1.0</span></td><td class="ok"><span class="v">1.0</span></td><td class="b4"><span class="v">0.844</span><span class="n">n=10000</span></td></tr>
<tr><th><code>dzasum</code></th><td class="ok"><span class="v">1.29</span></td><td class="ok"><span class="v">1.01</span></td><td class="ok"><span class="v">1.01</span></td></tr>
<tr><th><code>dznrm2</code></th><td class="ok"><span class="v">1.88</span></td><td class="ok"><span class="v">1.68</span></td><td class="ok"><span class="v">1.74</span></td></tr>
<tr><th><code>iamax</code></th><td class="b2"><span class="v">0.97</span><span class="n">n=3000</span></td><td class="ok"><span class="v">1.0</span></td><td class="b2"><span class="v">0.958</span><span class="n">n=1000000</span></td></tr>
<tr><th><code>izamax</code></th><td class="ok"><span class="v">1.08</span></td><td class="ok"><span class="v">1.32</span></td><td class="ok"><span class="v">1.51</span></td></tr>
<tr><th><code>nrm2</code></th><td class="ok"><span class="v">1.86</span></td><td class="ok"><span class="v">2.01</span></td><td class="ok"><span class="v">1.88</span></td></tr>
<tr><th><code>scal</code></th><td class="ok"><span class="v">1.0</span></td><td class="b1"><span class="v">0.994</span><span class="n">n=30000</span></td><td class="b3"><span class="v">0.948</span><span class="n">n=1000000</span></td></tr>
<tr><th><code>zaxpy</code></th><td class="b2"><span class="v">0.962</span><span class="n">n=1000000</span></td><td class="b2"><span class="v">0.987</span><span class="n">n=30000</span></td><td class="b3"><span class="v">0.939</span><span class="n">n=300000</span></td></tr>
<tr><th><code>zdotc</code></th><td class="ok"><span class="v">1.0</span></td><td class="ok"><span class="v">1.01</span></td><td class="b3"><span class="v">0.947</span><span class="n">n=300000</span></td></tr>
<tr><th><code>zdotu</code></th><td class="b1"><span class="v">0.993</span><span class="n">n=100000</span></td><td class="ok"><span class="v">1.01</span></td><td class="b3"><span class="v">0.908</span><span class="n">n=1000000</span></td></tr>
<tr><th><code>zscal</code></th><td class="ok"><span class="v">1.18</span></td><td class="b2"><span class="v">0.986</span><span class="n">n=100000</span></td><td class="b1"><span class="v">0.993</span><span class="n">n=300000</span></td></tr>
</tbody></table>
```

#### BLAS-2

```@raw html
<table class="pbg"><thead><tr><th>routine</th><th>Zen3 · AVX2</th><th>Zen4 · AVX-512</th><th>Zen5 · AVX-512</th></tr></thead><tbody>
<tr><th><code>gbmvN</code></th><td class="ok"><span class="v">1.01</span></td><td class="ok"><span class="v">1.42</span></td><td class="ok"><span class="v">1.25</span></td></tr>
<tr><th><code>gemvN</code></th><td class="b2"><span class="v">0.962</span><span class="n">n=50</span></td><td class="ok"><span class="v">1.0</span></td><td class="b4"><span class="v">0.782</span><span class="n">n=2100</span></td></tr>
<tr><th><code>gemvT</code></th><td class="b3"><span class="v">0.867</span><span class="n">n=2100</span></td><td class="b3"><span class="v">0.902</span><span class="n">n=2100</span></td><td class="b3"><span class="v">0.902</span><span class="n">n=2100</span></td></tr>
<tr><th><code>ger</code></th><td class="b2"><span class="v">0.987</span><span class="n">n=2100</span></td><td class="b2"><span class="v">0.979</span><span class="n">n=1024</span></td><td class="b3"><span class="v">0.949</span><span class="n">n=4096</span></td></tr>
<tr><th><code>sbmv</code></th><td class="ok"><span class="v">1.37</span></td><td class="ok"><span class="v">1.13</span></td><td class="ok"><span class="v">1.18</span></td></tr>
<tr><th><code>spmv</code></th><td class="ok"><span class="v">1.99</span></td><td class="ok"><span class="v">1.83</span></td><td class="ok"><span class="v">1.47</span></td></tr>
<tr><th><code>symv</code></th><td class="b3"><span class="v">0.913</span><span class="n">n=2100</span></td><td class="b3"><span class="v">0.941</span><span class="n">n=2100</span></td><td class="b2"><span class="v">0.966</span><span class="n">n=1024</span></td></tr>
<tr><th><code>trmv</code></th><td class="b2"><span class="v">0.959</span><span class="n">n=4096</span></td><td class="b3"><span class="v">0.918</span><span class="n">n=100</span></td><td class="b3"><span class="v">0.891</span><span class="n">n=100</span></td></tr>
<tr><th><code>trsv</code></th><td class="ok"><span class="v">1.0</span></td><td class="b2"><span class="v">0.987</span><span class="n">n=4096</span></td><td class="b1"><span class="v">0.993</span><span class="n">n=4096</span></td></tr>
<tr><th><code>trsvLN</code></th><td class="b2"><span class="v">0.987</span><span class="n">n=512</span></td><td class="b3"><span class="v">0.903</span><span class="n">n=100</span></td><td class="b3"><span class="v">0.944</span><span class="n">n=100</span></td></tr>
<tr><th><code>trsvLT</code></th><td class="b2"><span class="v">0.978</span><span class="n">n=4096</span></td><td class="b2"><span class="v">0.974</span><span class="n">n=2100</span></td><td class="b2"><span class="v">0.982</span><span class="n">n=4096</span></td></tr>
<tr><th><code>zgbmvN</code></th><td class="ok"><span class="v">1.16</span></td><td class="ok"><span class="v">1.03</span></td><td class="ok"><span class="v">1.02</span></td></tr>
<tr><th><code>zgemvC</code></th><td class="b3"><span class="v">0.932</span><span class="n">n=2100</span></td><td class="b1"><span class="v">0.99</span><span class="n">n=4096</span></td><td class="b2"><span class="v">0.976</span><span class="n">n=512</span></td></tr>
<tr><th><code>zgemvN</code></th><td class="b1"><span class="v">0.992</span><span class="n">n=4096</span></td><td class="b2"><span class="v">0.958</span><span class="n">n=512</span></td><td class="b3"><span class="v">0.909</span><span class="n">n=2048</span></td></tr>
<tr><th><code>zgemvT</code></th><td class="b3"><span class="v">0.938</span><span class="n">n=2100</span></td><td class="b2"><span class="v">0.988</span><span class="n">n=64</span></td><td class="b3"><span class="v">0.949</span><span class="n">n=64</span></td></tr>
<tr><th><code>zgeru</code></th><td class="b2"><span class="v">0.985</span><span class="n">n=1000</span></td><td class="b3"><span class="v">0.869</span><span class="n">n=1024</span></td><td class="b3"><span class="v">0.92</span><span class="n">n=100</span></td></tr>
<tr><th><code>zhbmv</code></th><td class="ok"><span class="v">1.33</span></td><td class="ok"><span class="v">1.05</span></td><td class="ok"><span class="v">1.04</span></td></tr>
<tr><th><code>zhemv</code></th><td class="b3"><span class="v">0.943</span><span class="n">n=4096</span></td><td class="ok"><span class="v">1.1</span></td><td class="ok"><span class="v">1.26</span></td></tr>
<tr><th><code>zhpmv</code></th><td class="ok"><span class="v">1.29</span></td><td class="ok"><span class="v">1.12</span></td><td class="ok"><span class="v">1.11</span></td></tr>
<tr><th><code>ztrmv</code></th><td class="b3"><span class="v">0.94</span><span class="n">n=256</span></td><td class="b2"><span class="v">0.987</span><span class="n">n=1024</span></td><td class="ok"><span class="v">1.06</span></td></tr>
<tr><th><code>ztrsv</code></th><td class="ok"><span class="v">1.02</span></td><td class="b3"><span class="v">0.925</span><span class="n">n=1024</span></td><td class="b3"><span class="v">0.947</span><span class="n">n=1024</span></td></tr>
</tbody></table>
```

#### BLAS-3

```@raw html
<table class="pbg"><thead><tr><th>routine</th><th>Zen3 · AVX2</th><th>Zen4 · AVX-512</th><th>Zen5 · AVX-512</th></tr></thead><tbody>
<tr><th><code>gemm</code></th><td class="b2"><span class="v">0.981</span><span class="n">n=1000</span></td><td class="b3"><span class="v">0.924</span><span class="n">n=100</span></td><td class="b3"><span class="v">0.898</span><span class="n">n=50</span></td></tr>
<tr><th><code>symm</code></th><td class="b2"><span class="v">0.956</span><span class="n">n=256</span></td><td class="b2"><span class="v">0.978</span><span class="n">n=1000</span></td><td class="ok"><span class="v">1.06</span></td></tr>
<tr><th><code>syr2k</code></th><td class="b3"><span class="v">0.948</span><span class="n">n=2100</span></td><td class="b3"><span class="v">0.92</span><span class="n">n=4096</span></td><td class="b2"><span class="v">0.956</span><span class="n">n=4096</span></td></tr>
<tr><th><code>syrk</code></th><td class="b3"><span class="v">0.88</span><span class="n">n=50</span></td><td class="b3"><span class="v">0.909</span><span class="n">n=4096</span></td><td class="b3"><span class="v">0.942</span><span class="n">n=100</span></td></tr>
<tr><th><code>trmm</code></th><td class="b2"><span class="v">0.961</span><span class="n">n=1000</span></td><td class="b3"><span class="v">0.947</span><span class="n">n=4096</span></td><td class="b2"><span class="v">0.987</span><span class="n">n=2100</span></td></tr>
<tr><th><code>trmmR</code></th><td class="b3"><span class="v">0.922</span><span class="n">n=256</span></td><td class="b3"><span class="v">0.937</span><span class="n">n=50</span></td><td class="b2"><span class="v">0.971</span><span class="n">n=1000</span></td></tr>
<tr><th><code>trsm</code></th><td class="b2"><span class="v">0.986</span><span class="n">n=1000</span></td><td class="b3"><span class="v">0.905</span><span class="n">n=100</span></td><td class="b3"><span class="v">0.946</span><span class="n">n=100</span></td></tr>
<tr><th><code>trsmR</code></th><td class="b2"><span class="v">0.957</span><span class="n">n=32</span></td><td class="b3"><span class="v">0.94</span><span class="n">n=1000</span></td><td class="b2"><span class="v">0.977</span><span class="n">n=1000</span></td></tr>
<tr><th><code>zgemm</code></th><td class="b2"><span class="v">0.975</span><span class="n">n=32</span></td><td class="b3"><span class="v">0.93</span><span class="n">n=50</span></td><td class="b2"><span class="v">0.98</span><span class="n">n=32</span></td></tr>
<tr><th><code>zhemm</code></th><td class="ok"><span class="v">1.04</span></td><td class="ok"><span class="v">1.0</span></td><td class="ok"><span class="v">1.17</span></td></tr>
<tr><th><code>zher2k</code></th><td class="b2"><span class="v">0.967</span><span class="n">n=128</span></td><td class="b2"><span class="v">0.981</span><span class="n">n=50</span></td><td class="ok"><span class="v">1.0</span></td></tr>
<tr><th><code>zherk</code></th><td class="b2"><span class="v">0.984</span><span class="n">n=128</span></td><td class="ok"><span class="v">1.06</span></td><td class="ok"><span class="v">1.05</span></td></tr>
<tr><th><code>zsymm</code></th><td class="ok"><span class="v">1.04</span></td><td class="b2"><span class="v">0.977</span><span class="n">n=50</span></td><td class="ok"><span class="v">1.11</span></td></tr>
<tr><th><code>zsyr2k</code></th><td class="b2"><span class="v">0.973</span><span class="n">n=128</span></td><td class="b1"><span class="v">0.994</span><span class="n">n=50</span></td><td class="b3"><span class="v">0.932</span><span class="n">n=100</span></td></tr>
<tr><th><code>zsyrk</code></th><td class="ok"><span class="v">1.02</span></td><td class="ok"><span class="v">1.01</span></td><td class="b2"><span class="v">0.966</span><span class="n">n=32</span></td></tr>
<tr><th><code>ztrmm</code></th><td class="b2"><span class="v">0.963</span><span class="n">n=50</span></td><td class="b3"><span class="v">0.939</span><span class="n">n=128</span></td><td class="b3"><span class="v">0.926</span><span class="n">n=128</span></td></tr>
<tr><th><code>ztrmmR</code></th><td class="b2"><span class="v">0.989</span><span class="n">n=50</span></td><td class="b2"><span class="v">0.966</span><span class="n">n=50</span></td><td class="b2"><span class="v">0.981</span><span class="n">n=128</span></td></tr>
<tr><th><code>ztrsm</code></th><td class="b1"><span class="v">0.992</span><span class="n">n=50</span></td><td class="b2"><span class="v">0.989</span><span class="n">n=100</span></td><td class="ok"><span class="v">1.01</span></td></tr>
<tr><th><code>ztrsmR</code></th><td class="b3"><span class="v">0.926</span><span class="n">n=100</span></td><td class="ok"><span class="v">1.01</span></td><td class="b2"><span class="v">0.985</span><span class="n">n=100</span></td></tr>
</tbody></table>
```

#### LAPACK

```@raw html
<table class="pbg"><thead><tr><th>routine</th><th>Zen3 · AVX2</th><th>Zen4 · AVX-512</th><th>Zen5 · AVX-512</th></tr></thead><tbody>
<tr><th><code>gbtrf</code></th><td class="b3"><span class="v">0.921</span><span class="n">n=1000</span></td><td class="b2"><span class="v">0.965</span><span class="n">n=1000</span></td><td class="ok"><span class="v">1.0</span></td></tr>
<tr><th><code>gels</code></th><td class="ok"><span class="v">1.02</span></td><td class="ok"><span class="v">1.22</span></td><td class="ok"><span class="v">1.18</span></td></tr>
<tr><th><code>geqp3</code></th><td class="b3"><span class="v">0.851</span><span class="n">n=50</span></td><td class="b2"><span class="v">0.961</span><span class="n">n=50</span></td><td class="b2"><span class="v">0.952</span><span class="n">n=50</span></td></tr>
<tr><th><code>geqrf</code></th><td class="b3"><span class="v">0.938</span><span class="n">n=50</span></td><td class="b3"><span class="v">0.915</span><span class="n">n=32</span></td><td class="b3"><span class="v">0.949</span><span class="n">n=32</span></td></tr>
<tr><th><code>gesvd</code></th><td class="b3"><span class="v">0.895</span><span class="n">n=8</span></td><td class="b2"><span class="v">0.965</span><span class="n">n=1000</span></td><td class="b2"><span class="v">0.96</span><span class="n">n=1000</span></td></tr>
<tr><th><code>getrf</code></th><td class="ok"><span class="v">1.02</span></td><td class="b2"><span class="v">0.964</span><span class="n">n=2100</span></td><td class="b2"><span class="v">0.959</span><span class="n">n=32</span></td></tr>
<tr><th><code>getri</code></th><td class="b4"><span class="v">0.47</span><span class="n">n=8</span></td><td><span class="n">—</span></td><td><span class="n">—</span></td></tr>
<tr><th><code>getrs</code></th><td class="b2"><span class="v">0.954</span><span class="n">n=100</span></td><td class="b4"><span class="v">0.839</span><span class="n">n=100</span></td><td class="b4"><span class="v">0.832</span><span class="n">n=100</span></td></tr>
<tr><th><code>gtsv</code></th><td class="ok"><span class="v">1.2</span></td><td class="ok"><span class="v">1.21</span></td><td class="ok"><span class="v">1.2</span></td></tr>
<tr><th><code>gttrf</code></th><td class="ok"><span class="v">1.48</span></td><td class="ok"><span class="v">1.48</span></td><td class="ok"><span class="v">1.43</span></td></tr>
<tr><th><code>gttrs</code></th><td class="ok"><span class="v">1.0</span></td><td class="ok"><span class="v">1.0</span></td><td class="b1"><span class="v">0.992</span><span class="n">n=262144</span></td></tr>
<tr><th><code>pbtrfL</code></th><td class="ok"><span class="v">1.06</span></td><td class="ok"><span class="v">1.13</span></td><td class="ok"><span class="v">1.07</span></td></tr>
<tr><th><code>pbtrfU</code></th><td class="b3"><span class="v">0.913</span><span class="n">n=128</span></td><td class="ok"><span class="v">1.12</span></td><td class="b2"><span class="v">0.974</span><span class="n">n=256</span></td></tr>
<tr><th><code>potrf</code></th><td class="ok"><span class="v">1.04</span></td><td class="b2"><span class="v">0.957</span><span class="n">n=2100</span></td><td class="b3"><span class="v">0.904</span><span class="n">n=8</span></td></tr>
<tr><th><code>potrfU</code></th><td class="b3"><span class="v">0.926</span><span class="n">n=1000</span></td><td class="b3"><span class="v">0.918</span><span class="n">n=2100</span></td><td class="b3"><span class="v">0.934</span><span class="n">n=1000</span></td></tr>
<tr><th><code>potri</code></th><td class="b4"><span class="v">0.762</span><span class="n">n=8</span></td><td><span class="n">—</span></td><td><span class="n">—</span></td></tr>
<tr><th><code>potrsL</code></th><td class="ok"><span class="v">1.0</span></td><td class="ok"><span class="v">1.03</span></td><td class="ok"><span class="v">1.02</span></td></tr>
<tr><th><code>potrsU</code></th><td class="b2"><span class="v">0.979</span><span class="n">n=2048</span></td><td class="b2"><span class="v">0.956</span><span class="n">n=1024</span></td><td class="b3"><span class="v">0.941</span><span class="n">n=512</span></td></tr>
<tr><th><code>pptrfL</code></th><td class="ok"><span class="v">1.04</span></td><td class="ok"><span class="v">1.03</span></td><td class="ok"><span class="v">1.0</span></td></tr>
<tr><th><code>pptrfU</code></th><td class="ok"><span class="v">1.71</span></td><td class="ok"><span class="v">1.29</span></td><td class="ok"><span class="v">1.44</span></td></tr>
<tr><th><code>pstrf</code></th><td class="b2"><span class="v">0.964</span><span class="n">n=1000</span></td><td class="ok"><span class="v">1.01</span></td><td class="ok"><span class="v">1.06</span></td></tr>
<tr><th><code>pstrfU</code></th><td class="b4"><span class="v">0.231</span><span class="n">n=8</span></td><td class="b3"><span class="v">0.935</span><span class="n">n=2100</span></td><td class="b2"><span class="v">0.963</span><span class="n">n=50</span></td></tr>
<tr><th><code>ptsv</code></th><td class="ok"><span class="v">1.05</span></td><td class="ok"><span class="v">1.05</span></td><td class="ok"><span class="v">1.06</span></td></tr>
<tr><th><code>pttrf</code></th><td class="ok"><span class="v">1.11</span></td><td class="ok"><span class="v">1.12</span></td><td class="ok"><span class="v">1.09</span></td></tr>
<tr><th><code>pttrs</code></th><td class="b2"><span class="v">0.975</span><span class="n">n=256</span></td><td class="b2"><span class="v">0.957</span><span class="n">n=256</span></td><td class="b1"><span class="v">0.993</span><span class="n">n=1024</span></td></tr>
<tr><th><code>syev</code></th><td class="ok"><span class="v">1.09</span></td><td class="ok"><span class="v">1.07</span></td><td class="ok"><span class="v">1.05</span></td></tr>
<tr><th><code>syevN</code></th><td class="ok"><span class="v">1.03</span></td><td class="b2"><span class="v">0.976</span><span class="n">n=1024</span></td><td class="ok"><span class="v">1.08</span></td></tr>
<tr><th><code>sytrf</code></th><td class="ok"><span class="v">1.04</span></td><td class="b4"><span class="v">0.806</span><span class="n">n=4096</span></td><td class="b2"><span class="v">0.985</span><span class="n">n=50</span></td></tr>
<tr><th><code>sytrs</code></th><td class="ok"><span class="v">1.35</span></td><td class="ok"><span class="v">1.37</span></td><td class="ok"><span class="v">1.29</span></td></tr>
<tr><th><code>trtri</code></th><td class="ok"><span class="v">1.02</span></td><td><span class="n">—</span></td><td><span class="n">—</span></td></tr>
<tr><th><code>trtrs</code></th><td class="b2"><span class="v">0.963</span><span class="n">n=50</span></td><td class="b3"><span class="v">0.906</span><span class="n">n=50</span></td><td class="b3"><span class="v">0.929</span><span class="n">n=100</span></td></tr>
<tr><th><code>zgeqrf</code></th><td class="ok"><span class="v">1.13</span></td><td class="ok"><span class="v">1.02</span></td><td class="ok"><span class="v">1.07</span></td></tr>
<tr><th><code>zgesvd</code></th><td class="ok"><span class="v">1.02</span></td><td class="b2"><span class="v">0.972</span><span class="n">n=50</span></td><td class="ok"><span class="v">1.03</span></td></tr>
<tr><th><code>zgetrf</code></th><td class="ok"><span class="v">1.0</span></td><td class="b4"><span class="v">0.826</span><span class="n">n=50</span></td><td class="b3"><span class="v">0.862</span><span class="n">n=100</span></td></tr>
<tr><th><code>zheev</code></th><td class="ok"><span class="v">1.21</span></td><td class="ok"><span class="v">1.11</span></td><td class="ok"><span class="v">1.07</span></td></tr>
<tr><th><code>zheevN</code></th><td class="ok"><span class="v">1.12</span></td><td class="ok"><span class="v">1.08</span></td><td class="ok"><span class="v">1.15</span></td></tr>
<tr><th><code>zpotrf</code></th><td class="ok"><span class="v">1.02</span></td><td class="ok"><span class="v">1.14</span></td><td class="ok"><span class="v">1.13</span></td></tr>
<tr><th><code>zpotrfU</code></th><td class="ok"><span class="v">1.04</span></td><td class="ok"><span class="v">1.02</span></td><td class="ok"><span class="v">1.04</span></td></tr>
</tbody></table>
```
```@raw html
<p class="pbg-key">
 <span><i style="background:#1f8a5b"></i>gates (≥ 1.0)</span>
 <span><i style="background:#7a8496"></i>≥ 0.99</span>
 <span><i style="background:#c07d12"></i>≥ 0.95</span>
 <span><i style="background:#cf5a35"></i>≥ 0.85</span>
 <span><i style="background:#b3243a"></i>below 0.85</span>
</p>
```

## LAPACK — factorizations & solves

| Op | Routines | Types | Routes | Zen3 | Zen4 | Zen5 | Zen4 vs OB geo/worst | Zen4 vs AOCL geo/worst |
|---|---|---|---|---|---|---|---|---|
| Cholesky (lower) | potrf | s/d/c/z | ✅ | 1.04 | **0.957** | **0.903** | 1.49 / 0.99 | 1.57 / 0.957 |
| Cholesky (upper) | potrf `uplo='U'` | s/d/c/z | ✅ | **0.926** | **0.919** | **0.935** | 1.42 / 1.1 | 1.36 / 0.919 |
| Cholesky solve | potrs | s/d/c/z | ✅ | **0.972** | **0.918** | **0.939** | 2.83 / 1.19 | 1.23 / 0.918 |
| Pivoted Cholesky | pstrf | s/d/c/z | ✅ | **0.237** | **0.935** | **0.962** | 1.29 / 0.935 | 1.24 / 0.987 |
| LU | getrf, gesv | s/d/c/z | ✅ | 1.02 | **0.964** | **0.958** | 1.3 / 1.02 | 1.46 / 0.964 |
| LU solve | getrs | s/d/c/z | ✅ | **0.951** | **0.839** | **0.833** | 1.24 / 1.04 | 1.1 / 0.839 |
| QR | geqrf, orgqr, ormqr | s/d/c/z | ✅ | **0.934** | **0.917** | **0.946** | 1.79 / 1.19 | 1.46 / 0.917 |
| Pivoted QR | geqp3 | s/d/c/z | ✅ | **0.851** | **0.96** | **0.952** | 1.3 / 1.02 | 1.14 / 0.96 |
| Bunch–Kaufman | sytrf, hetrf | s/d/c/z | ✅ | 1.03 | **0.799** | **0.983** | 1.36 / 0.859 | 1.33 / 0.799 |
| Bunch–Kaufman solve | sytrs, hetrs | s/d/c/z | ✅ | 1.35 | 1.36 | 1.28 | 1.7 / 1.47 | 1.96 / 1.36 |
| Triangular solve | trtrs | s/d/c/z | ✅ | **0.953** | **0.906** | **0.93** | 1.2 / 0.954 | 1.14 / 0.906 |
| Least-squares | gels | s/d/c/z | ✅ | 1.01 | 1.22 | 1.19 | 2.34 / 1.36 | 1.77 / 1.22 |
| SVD | gesvd, gesdd | s/d/c/z | ✅ | 1.03 | **0.964** | **0.96** | 1.23 / 1.05 | 1.18 / 0.964 |
| Symmetric eigen | syev, syevd, syevr | s/d/c/z | ✅ | 1.03 | **0.976** | 1.05 | 1.36 / 0.976 | 1.5 / 1.07 |

The `geo/worst` columns are per reference and scoped to Zen4; the per-box columns left of them are the
gate. The solves (`potrs`/`getrs`/`trtrs`) are not yet gated — the band on those rows is for the
factorization.

## LAPACK — SVD

| Op | Routines | Types | Routes | Zen3 | Zen4 | Zen5 |
|---|---|---|---|---|---|---|
| SVD complex | gesvd, gesdd (z/c) | c/z | ✅ | | | ⏳ |
| Generalized SVD | ggsvd, ggsvd3 | s/d/c/z | ✅ (rank-deficient) | | | ⏳ |

## LAPACK — eigensolvers

| Op | Routines | Types | Routes | Zen3 | Zen4 | Zen5 |
|---|---|---|---|---|---|---|
| Symmetric / Hermitian (vectors) | syev, syevd, syevr, heev, sytrd, hetrd, ormtr, *stedc\*, steqr\** | s/d/c/z | ✅ | 1.09 | 1.07 | 1.05 |
| Symmetric / Hermitian (values only) | syev, *sterf\** | s/d/c/z | ✅ | 1.03 | **0.976** | 1.08 |
| Sym-tridiagonal | stev, stegr, stebz, stein | s/d | ✅ | | | ⏳ |
| Generalized symmetric | sygvd, hegvd | s/d/c/z | ✅ | | | ⏳ |
| Nonsymmetric | geev, geevx, gebal, gehrd, hseqr, trevc, gebak | s/d/c/z | ✅ | | | ⏳ |
| Schur | gees | s/d/c/z | ✅ | | | ⏳ |
| Generalized nonsym (QZ) | ggev, gges, gghrd, hgeqz, tgevc | s/d/c/z | ✅ | | | ⏳ |
| Schur reordering | trexc, trsen | s/d/c/z | ✅ | | | ⏳ |
| Sylvester / Lyapunov | trsyl | s/d/c/z | ✅ | | | ⏳ |

*\* `stedc`, `steqr` and `sterf` are **implemented but not forwarded**: they are the internal
building blocks `syev`/`heev` are composed from, and PureBLAS exposes no `stedc_64_`/`steqr_64_`/
`sterf_64_` symbol, so a program calling one of them **directly** still reaches OpenBLAS. Every other
routine in these two rows is forwarded and does route to PureBLAS. Verified against `src/cabi/`,
2026-08-20.*

## LAPACK — banded / tridiagonal / packed

| Op | Routines | Types | Routes | Zen3 | Zen4 | Zen5 | Zen4 vs OB geo/worst | Zen4 vs AOCL geo/worst |
|---|---|---|---|---|---|---|---|---|
| General banded LU | gbtrf, gbtrs | s/d/c/z | ✅ | **0.921** | **0.965** | 1 | 1.74 / 1.26 | 1.37 / 0.965 |
| General tridiagonal | gtsv, gttrf, gttrs | s/d/c/z | ✅ | 1 | 1 | **0.992** | 1.34 / 1 | 1.27 / 1.03 |
| SPD tridiagonal | pttrf, pttrs, ptsv | s/d/c/z | ✅ | **0.975** | **0.957** | **0.994** | 1.38 / 1.12 | 1.06 / 0.957 |
| Banded Cholesky | pbtrf, pbtrs | s/d/c/z | ✅ | **0.913** | 1.12 | **0.974** | 1.59 / 1.21 | 1.53 / 1.12 |
| Packed Cholesky | pptrf, pptrs | s/d/c/z | ✅ | 1.05 | 1.02 | 1 | 2.38 / 1.29 | 3.75 / 1.02 |

`pttrs`'s 0.99 vs AOCL is a shared dependency-chain bound, not a gap — see [Notes](notes.md).

## Free via composition

`exp`, `sqrt`, `log`, `^` of a matrix, `sylvester`/`lyap`, `pinv`, `nullspace`,
`rank`, `cond`, `factorize` — computed in Julia on top of the routed
`eigen`/`schur`/`svd`/`\` kernels; no separate LAPACK wrapper needed.

## OpenBLAS fallthrough: ZERO

**Every LAPACK symbol `LinearAlgebra` can `ccall` now forwards to PureBLAS after
`activate()`** — including the auxiliaries (`larf`/`larfg`/`lacpy`), the driver internals
(`gebrd`/`bdsqr`/`bdsdc`/`hseqr`/`trevc`/`gebak`/`sytrd`/`hetrd`/`orgtr`/`ormtr`), the
combined and expert drivers (`gesv`, `posv`, **`gesvx`** with equilibration + iterative
refinement + condition/error bounds), the reordering routines (`trexc`/`trsen`/**`tgsen`**,
real *and* complex — the real path does the 2×2 conjugate-pair swap), **`trrfs`**,
**`syconv`**, complex **`bdsqr`**, and the **rank-deficient generalized SVD** (`ggsvd`,
all s/d/c/z). This is enforced by a machine-checkable ratchet test (`test/lbt_forward_tests.jl`)
that enumerates every symbol the stdlib wraps and asserts the fallthrough count is **0**.

The only two names excluded from the count are `cstev_`/`zstev_`, which are **not real LAPACK
symbols** — they appear only in commented-out lines of the stdlib and have no OpenBLAS export.

## Summary

- **Routing is complete** — every operation in the tables reaches PureBLAS after `activate()`, and every
  one is numerically LAPACK-accurate. The ratchet gate above confirms zero OpenBLAS fallthrough.
- **Performance is typically well ahead of both references but not yet uniformly so.** On the Zen4 sweep
  the geomeans run ~1.0–2.4× across BLAS and the dense factorizations, and 36 of the 83 measured rows
  clear `≥ max(OpenBLAS, AOCL)` at *every* size. The rest miss somewhere, usually narrowly. Which cells
  and why: [Notes](notes.md).
- **The second tier** — all eigensolvers (symmetric, Hermitian, nonsymmetric, generalized, Schur),
  Sylvester/Schur-reordering, `gesvx`, generalized SVD, and the remaining factorizations (indefinite,
  QL/RQ, RZ, pivoted Cholesky, banded/tridiagonal/packed, rank-deficient LS) — is routed and numerically
  LAPACK-accurate, correctness-first; its `≥ max(OpenBLAS, AOCL)` gate is a scheduled follow-up campaign.
