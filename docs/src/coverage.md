# LAPACK / BLAS coverage

Which `LinearAlgebra` operations route to PureBLAS after `PureBLAS.activate()`, and how fast they are.

Every number is a PB / max(OpenBLAS, AOCL) speed ratio — the routine's worst (op, size) cell on that
box, one column per microarchitecture. A ratio of 1.0 or better, to two significant digits, gates, so
0.995 passes and 0.9949 does not. A miss is printed at three digits and floored, then bolded in the
routing tables or colour-banded in the BLAS ones. "Routes" means the call forwards to PureBLAS via LBT
after `activate()`, and ⏳ marks a routine that is not gated yet. Types are s = Float32, d = Float64,
c = ComplexF32, z = ComplexF64.

How the measurements are taken and how to read a cell: [Methodology](methodology.md). Per-routine
analysis and history: [Notes](notes.md).

## BLAS

One row per routine, generated from the caches by `bench/coverage_ops.jl`; `n=` names the worst cell.

```@raw html
<style>
.pbg{--l:#e3e6ee;--m:#5d6675;--ok:#1f8a5b;--b1:#7a8496;--b2:#c07d12;--b3:#cf5a35;--b4:#b3243a;
 --okbg:#e9f6ef;--b1bg:#f1f3f7;--b2bg:#fdf3e2;--b3bg:#fceee9;--b4bg:#fbe9ed;
 border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;display:table}
/* The scroll container. Vitepress gets wide tables to scroll with `.vp-doc table{display:block;
   overflow-x:auto}`; `.pbg` needs display:table for column sizing, and setting it removes that
   container. Every th/td here is white-space:nowrap, so the table's min-content width exceeds a
   narrow content column and it then overflows into the page outline instead of scrolling. Wrap
   rather than drop the nowrap: a wrapped ratio cell is worse to read than a scrollbar. */
.pbg-wrap{overflow-x:auto;margin:20px 0;max-width:100%}
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
/* provenance column: the revision a row's numbers were measured at. `hx` (mixed) is deliberately loud —
   a row whose boxes were swept at different commits is not a uniform measurement and must not read as one. */
.pbg td.h .n{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;margin-left:0}
.pbg td.hx{background:var(--b3bg)} .pbg td.hx .n{color:var(--b3);font-weight:600;margin-left:0}
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
<div class="pbg-wrap"><table class="pbg"><thead><tr><th>routine</th><th>Zen3 · AVX2</th><th>Zen4 · AVX-512</th><th>Zen5 · AVX-512</th><th>swept at</th><th>toolchain</th></tr></thead><tbody>
<tr><th><code>asum</code></th><td class="ok"><span class="v">1.29</span></td><td class="ok"><span class="v">1.0</span></td><td class="b3"><span class="v">0.92</span><span class="n">n=1000000</span></td><td class="h"><span class="n">5647c65</span></td><td class="h"><span class="n">1.13.0-rc4</span></td></tr>
<tr><th><code>axpy</code></th><td class="ok"><span class="v">1.0</span></td><td class="ok"><span class="v">1.01</span></td><td class="b2"><span class="v">0.954</span><span class="n">n=300000</span></td><td class="h"><span class="n">5647c65</span></td><td class="h"><span class="n">1.13.0-rc4</span></td></tr>
<tr><th><code>dot</code></th><td class="b1"><span class="v">0.991</span><span class="n">n=1000000</span></td><td class="ok"><span class="v">1.0</span></td><td class="b2"><span class="v">0.982</span><span class="n">n=1000000</span></td><td class="h"><span class="n">5647c65</span></td><td class="h"><span class="n">1.13.0-rc4</span></td></tr>
<tr><th><code>dzasum</code></th><td class="ok"><span class="v">1.26</span></td><td class="ok"><span class="v">1.0</span></td><td class="b2"><span class="v">0.986</span><span class="n">n=30000</span></td><td class="hx"><span class="n">⚠ mixed</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>dznrm2</code></th><td class="ok"><span class="v">1.84</span></td><td class="ok"><span class="v">1.66</span></td><td class="ok"><span class="v">1.79</span></td><td class="hx"><span class="n">⚠ mixed</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>iamax</code></th><td class="b1"><span class="v">0.994</span><span class="n">n=100000</span></td><td class="ok"><span class="v">1.0</span></td><td class="b2"><span class="v">0.974</span><span class="n">n=1000000</span></td><td class="h"><span class="n">5647c65</span></td><td class="h"><span class="n">1.13.0-rc4</span></td></tr>
<tr><th><code>izamax</code></th><td class="ok"><span class="v">1.06</span></td><td class="ok"><span class="v">1.46</span></td><td class="ok"><span class="v">1.72</span></td><td class="hx"><span class="n">⚠ mixed</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>nrm2</code></th><td class="ok"><span class="v">1.91</span></td><td class="ok"><span class="v">2.01</span></td><td class="ok"><span class="v">1.97</span></td><td class="h"><span class="n">5647c65</span></td><td class="h"><span class="n">1.13.0-rc4</span></td></tr>
<tr><th><code>scal</code></th><td class="b1"><span class="v">0.994</span><span class="n">n=300000</span></td><td class="b1"><span class="v">0.994</span><span class="n">n=30000</span></td><td class="b2"><span class="v">0.985</span><span class="n">n=1000000</span></td><td class="h"><span class="n">5647c65</span></td><td class="h"><span class="n">1.13.0-rc4</span></td></tr>
<tr><th><code>zaxpy</code></th><td class="ok"><span class="v">1.0</span></td><td class="b2"><span class="v">0.963</span><span class="n">n=30000</span></td><td class="b4"><span class="v">0.84</span><span class="n">n=30000</span></td><td class="hx"><span class="n">⚠ mixed</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zdotc</code></th><td class="ok"><span class="v">1.0</span></td><td class="ok"><span class="v">1.01</span></td><td class="b3"><span class="v">0.922</span><span class="n">n=300000</span></td><td class="hx"><span class="n">⚠ mixed</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zdotu</code></th><td class="ok"><span class="v">1.0</span></td><td class="ok"><span class="v">1.01</span></td><td class="b3"><span class="v">0.939</span><span class="n">n=300000</span></td><td class="hx"><span class="n">⚠ mixed</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zscal</code></th><td class="ok"><span class="v">1.1</span></td><td class="b1"><span class="v">0.992</span><span class="n">n=300000</span></td><td class="b2"><span class="v">0.973</span><span class="n">n=1000000</span></td><td class="hx"><span class="n">⚠ mixed</span></td><td class="n"><span class="n">?</span></td></tr>
</tbody></table></div>
```

#### BLAS-2

```@raw html
<div class="pbg-wrap"><table class="pbg"><thead><tr><th>routine</th><th>Zen3 · AVX2</th><th>Zen4 · AVX-512</th><th>Zen5 · AVX-512</th><th>swept at</th><th>toolchain</th></tr></thead><tbody>
<tr><th><code>gbmvN</code></th><td class="ok"><span class="v">1.0</span></td><td class="ok"><span class="v">1.41</span></td><td class="ok"><span class="v">1.25</span></td><td class="h"><span class="n">5647c65</span></td><td class="h"><span class="n">1.13.0-rc4</span></td></tr>
<tr><th><code>gemvN</code></th><td class="b3"><span class="v">0.94</span><span class="n">n=4096</span></td><td class="b2"><span class="v">0.983</span><span class="n">n=100</span></td><td class="b4"><span class="v">0.787</span><span class="n">n=2100</span></td><td class="h"><span class="n">5647c65</span></td><td class="h"><span class="n">1.13.0-rc4</span></td></tr>
<tr><th><code>gemvT</code></th><td class="b3"><span class="v">0.87</span><span class="n">n=2100</span></td><td class="b3"><span class="v">0.892</span><span class="n">n=2100</span></td><td class="b3"><span class="v">0.9</span><span class="n">n=2100</span></td><td class="h"><span class="n">5647c65</span></td><td class="h"><span class="n">1.13.0-rc4</span></td></tr>
<tr><th><code>ger</code></th><td class="b2"><span class="v">0.97</span><span class="n">n=2048</span></td><td class="b2"><span class="v">0.981</span><span class="n">n=1000</span></td><td class="b2"><span class="v">0.96</span><span class="n">n=4096</span></td><td class="h"><span class="n">5647c65</span></td><td class="h"><span class="n">1.13.0-rc4</span></td></tr>
<tr><th><code>sbmv</code></th><td class="ok"><span class="v">1.38</span></td><td class="ok"><span class="v">1.12</span></td><td class="ok"><span class="v">1.18</span></td><td class="h"><span class="n">5647c65</span></td><td class="h"><span class="n">1.13.0-rc4</span></td></tr>
<tr><th><code>spmv</code></th><td class="ok"><span class="v">1.96</span></td><td class="ok"><span class="v">1.82</span></td><td class="ok"><span class="v">1.48</span></td><td class="h"><span class="n">5647c65</span></td><td class="h"><span class="n">1.13.0-rc4</span></td></tr>
<tr><th><code>symv</code></th><td class="b3"><span class="v">0.935</span><span class="n">n=2100</span></td><td class="b3"><span class="v">0.937</span><span class="n">n=2100</span></td><td class="b2"><span class="v">0.98</span><span class="n">n=2100</span></td><td class="h"><span class="n">5647c65</span></td><td class="h"><span class="n">1.13.0-rc4</span></td></tr>
<tr><th><code>trmv</code></th><td class="b2"><span class="v">0.968</span><span class="n">n=4096</span></td><td class="b3"><span class="v">0.935</span><span class="n">n=100</span></td><td class="b3"><span class="v">0.891</span><span class="n">n=100</span></td><td class="h"><span class="n">5647c65</span></td><td class="h"><span class="n">1.13.0-rc4</span></td></tr>
<tr><th><code>trsv</code></th><td class="ok"><span class="v">1.0</span></td><td class="b2"><span class="v">0.977</span><span class="n">n=512</span></td><td class="b1"><span class="v">0.99</span><span class="n">n=512</span></td><td class="h"><span class="n">5647c65</span></td><td class="h"><span class="n">1.13.0-rc4</span></td></tr>
<tr><th><code>trsvLN</code></th><td class="b2"><span class="v">0.975</span><span class="n">n=256</span></td><td class="b3"><span class="v">0.938</span><span class="n">n=100</span></td><td class="ok"><span class="v">1.0</span></td><td class="h"><span class="n">5647c65</span></td><td class="h"><span class="n">1.13.0-rc4</span></td></tr>
<tr><th><code>trsvLT</code></th><td class="ok"><span class="v">1.0</span></td><td class="b2"><span class="v">0.964</span><span class="n">n=2100</span></td><td class="ok"><span class="v">1.0</span></td><td class="h"><span class="n">5647c65</span></td><td class="h"><span class="n">1.13.0-rc4</span></td></tr>
<tr><th><code>zgbmvN</code></th><td class="ok"><span class="v">1.18</span></td><td class="ok"><span class="v">1.04</span></td><td class="ok"><span class="v">1.01</span></td><td class="hx"><span class="n">⚠ mixed</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zgemvC</code></th><td class="b2"><span class="v">0.954</span><span class="n">n=1024</span></td><td class="ok"><span class="v">1.0</span></td><td class="b3"><span class="v">0.9</span><span class="n">n=512</span></td><td class="hx"><span class="n">⚠ mixed</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zgemvN</code></th><td class="ok"><span class="v">1.0</span></td><td class="b2"><span class="v">0.976</span><span class="n">n=512</span></td><td class="b3"><span class="v">0.896</span><span class="n">n=512</span></td><td class="hx"><span class="n">⚠ mixed</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zgemvT</code></th><td class="b3"><span class="v">0.941</span><span class="n">n=1024</span></td><td class="ok"><span class="v">1.0</span></td><td class="b3"><span class="v">0.906</span><span class="n">n=512</span></td><td class="hx"><span class="n">⚠ mixed</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zgeru</code></th><td class="b2"><span class="v">0.964</span><span class="n">n=512</span></td><td class="b4"><span class="v">0.849</span><span class="n">n=1024</span></td><td class="b3"><span class="v">0.891</span><span class="n">n=1000</span></td><td class="hx"><span class="n">⚠ mixed</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zhbmv</code></th><td class="ok"><span class="v">1.34</span></td><td class="ok"><span class="v">1.05</span></td><td class="ok"><span class="v">1.03</span></td><td class="hx"><span class="n">⚠ mixed</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zhemv</code></th><td class="b3"><span class="v">0.932</span><span class="n">n=2048</span></td><td class="ok"><span class="v">1.04</span></td><td class="ok"><span class="v">1.23</span></td><td class="hx"><span class="n">⚠ mixed</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zhpmv</code></th><td class="ok"><span class="v">1.29</span></td><td class="ok"><span class="v">1.13</span></td><td class="ok"><span class="v">1.12</span></td><td class="hx"><span class="n">⚠ mixed</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>ztrmv</code></th><td class="ok"><span class="v">1.03</span></td><td class="b3"><span class="v">0.931</span><span class="n">n=1024</span></td><td class="b1"><span class="v">0.994</span><span class="n">n=1024</span></td><td class="hx"><span class="n">⚠ mixed</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>ztrsv</code></th><td class="b1"><span class="v">0.994</span><span class="n">n=256</span></td><td class="b3"><span class="v">0.93</span><span class="n">n=1024</span></td><td class="b2"><span class="v">0.954</span><span class="n">n=1000</span></td><td class="hx"><span class="n">⚠ mixed</span></td><td class="n"><span class="n">?</span></td></tr>
</tbody></table></div>
```

#### BLAS-3

```@raw html
<div class="pbg-wrap"><table class="pbg"><thead><tr><th>routine</th><th>Zen3 · AVX2</th><th>Zen4 · AVX-512</th><th>Zen5 · AVX-512</th><th>swept at</th><th>toolchain</th></tr></thead><tbody>
<tr><th><code>gemm</code></th><td class="b2"><span class="v">0.979</span><span class="n">n=1000</span></td><td class="b3"><span class="v">0.921</span><span class="n">n=100</span></td><td class="b3"><span class="v">0.905</span><span class="n">n=50</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>symm</code></th><td class="b2"><span class="v">0.952</span><span class="n">n=256</span></td><td class="b2"><span class="v">0.972</span><span class="n">n=1000</span></td><td class="ok"><span class="v">1.06</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>syr2k</code></th><td class="b3"><span class="v">0.946</span><span class="n">n=256</span></td><td class="b3"><span class="v">0.93</span><span class="n">n=4096</span></td><td class="b2"><span class="v">0.962</span><span class="n">n=4096</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>syrk</code></th><td class="b3"><span class="v">0.947</span><span class="n">n=2100</span></td><td class="b3"><span class="v">0.917</span><span class="n">n=4096</span></td><td class="b3"><span class="v">0.941</span><span class="n">n=100</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>trmm</code></th><td class="b2"><span class="v">0.962</span><span class="n">n=1000</span></td><td class="b3"><span class="v">0.942</span><span class="n">n=4096</span></td><td class="b2"><span class="v">0.981</span><span class="n">n=2100</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>trmmR</code></th><td class="b3"><span class="v">0.913</span><span class="n">n=256</span></td><td class="b3"><span class="v">0.907</span><span class="n">n=50</span></td><td class="b2"><span class="v">0.968</span><span class="n">n=1000</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>trsm</code></th><td class="b2"><span class="v">0.98</span><span class="n">n=256</span></td><td class="b3"><span class="v">0.902</span><span class="n">n=100</span></td><td class="b3"><span class="v">0.943</span><span class="n">n=100</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>trsmR</code></th><td class="b3"><span class="v">0.947</span><span class="n">n=128</span></td><td class="b3"><span class="v">0.918</span><span class="n">n=1000</span></td><td class="b2"><span class="v">0.959</span><span class="n">n=1000</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zgemm</code></th><td class="b2"><span class="v">0.977</span><span class="n">n=32</span></td><td class="b3"><span class="v">0.88</span><span class="n">n=50</span></td><td class="b2"><span class="v">0.964</span><span class="n">n=50</span></td><td class="h"><span class="n">7b4942c</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zhemm</code></th><td class="b2"><span class="v">0.966</span><span class="n">n=50</span></td><td class="ok"><span class="v">1.09</span></td><td class="ok"><span class="v">1.2</span></td><td class="h"><span class="n">7b4942c</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zher2k</code></th><td class="ok"><span class="v">1.01</span></td><td class="b3"><span class="v">0.897</span><span class="n">n=50</span></td><td class="b2"><span class="v">0.959</span><span class="n">n=50</span></td><td class="h"><span class="n">7b4942c</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zherk</code></th><td class="ok"><span class="v">1.04</span></td><td class="ok"><span class="v">1.01</span></td><td class="ok"><span class="v">1.02</span></td><td class="h"><span class="n">7b4942c</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zsymm</code></th><td class="b3"><span class="v">0.945</span><span class="n">n=50</span></td><td class="ok"><span class="v">1.08</span></td><td class="ok"><span class="v">1.19</span></td><td class="h"><span class="n">7b4942c</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zsyr2k</code></th><td class="b2"><span class="v">0.984</span><span class="n">n=128</span></td><td class="b3"><span class="v">0.907</span><span class="n">n=50</span></td><td class="b3"><span class="v">0.908</span><span class="n">n=100</span></td><td class="h"><span class="n">7b4942c</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zsyrk</code></th><td class="ok"><span class="v">1.02</span></td><td class="ok"><span class="v">1.0</span></td><td class="b3"><span class="v">0.934</span><span class="n">n=32</span></td><td class="h"><span class="n">7b4942c</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>ztrmm</code></th><td class="b2"><span class="v">0.973</span><span class="n">n=50</span></td><td class="b3"><span class="v">0.947</span><span class="n">n=128</span></td><td class="b3"><span class="v">0.912</span><span class="n">n=128</span></td><td class="h"><span class="n">7b4942c</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>ztrmmR</code></th><td class="b2"><span class="v">0.974</span><span class="n">n=50</span></td><td class="b1"><span class="v">0.994</span><span class="n">n=50</span></td><td class="b2"><span class="v">0.968</span><span class="n">n=128</span></td><td class="h"><span class="n">7b4942c</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>ztrsm</code></th><td class="b2"><span class="v">0.97</span><span class="n">n=50</span></td><td class="ok"><span class="v">1.0</span></td><td class="ok"><span class="v">1.02</span></td><td class="h"><span class="n">7b4942c</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>ztrsmR</code></th><td class="b3"><span class="v">0.855</span><span class="n">n=100</span></td><td class="b2"><span class="v">0.982</span><span class="n">n=50</span></td><td class="b2"><span class="v">0.962</span><span class="n">n=50</span></td><td class="h"><span class="n">7b4942c</span></td><td class="n"><span class="n">?</span></td></tr>
</tbody></table></div>
```

#### LAPACK

```@raw html
<div class="pbg-wrap"><table class="pbg"><thead><tr><th>routine</th><th>Zen3 · AVX2</th><th>Zen4 · AVX-512</th><th>Zen5 · AVX-512</th><th>swept at</th><th>toolchain</th></tr></thead><tbody>
<tr><th><code>gbtrf</code></th><td class="ok"><span class="v">1.03</span></td><td class="b2"><span class="v">0.988</span><span class="n">n=2048</span></td><td class="b1"><span class="v">0.993</span><span class="n">n=2048</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>gbtrs</code></th><td><span class="n">—</span></td><td class="ok"><span class="v">1.03</span></td><td><span class="n">—</span></td><td class="h"><span class="n">42563f8</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>geev</code></th><td class="b3"><span class="v">0.945</span><span class="n">n=256</span></td><td class="b3"><span class="v">0.943</span><span class="n">n=256</span></td><td class="b3"><span class="v">0.944</span><span class="n">n=256</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>gels</code></th><td class="ok"><span class="v">1.06</span></td><td class="ok"><span class="v">1.22</span></td><td class="ok"><span class="v">1.19</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>gelsd</code></th><td class="ok"><span class="v">1.32</span></td><td class="ok"><span class="v">1.27</span></td><td class="ok"><span class="v">1.27</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>gelsy</code></th><td class="ok"><span class="v">1.02</span></td><td class="b2"><span class="v">0.973</span><span class="n">n=1000</span></td><td class="ok"><span class="v">1.0</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>geqp3</code></th><td class="b3"><span class="v">0.898</span><span class="n">n=50</span></td><td class="b2"><span class="v">0.955</span><span class="n">n=1000</span></td><td class="ok"><span class="v">1.0</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>geqrf</code></th><td class="b2"><span class="v">0.955</span><span class="n">n=50</span></td><td class="b3"><span class="v">0.929</span><span class="n">n=32</span></td><td class="b3"><span class="v">0.944</span><span class="n">n=32</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>gesvd</code></th><td class="ok"><span class="v">1.04</span></td><td class="b2"><span class="v">0.957</span><span class="n">n=1000</span></td><td class="b2"><span class="v">0.965</span><span class="n">n=1000</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>getrf</code></th><td class="ok"><span class="v">1.01</span></td><td class="b3"><span class="v">0.939</span><span class="n">n=32</span></td><td class="b2"><span class="v">0.953</span><span class="n">n=32</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>getri</code></th><td class="b4"><span class="v">0.683</span><span class="n">n=8</span></td><td class="b3"><span class="v">0.919</span><span class="n">n=1000</span></td><td class="b2"><span class="v">0.954</span><span class="n">n=1000</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>getrs</code></th><td class="b2"><span class="v">0.962</span><span class="n">n=100</span></td><td class="b4"><span class="v">0.831</span><span class="n">n=100</span></td><td class="b4"><span class="v">0.82</span><span class="n">n=100</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>gtsv</code></th><td class="ok"><span class="v">1.2</span></td><td class="ok"><span class="v">1.2</span></td><td class="ok"><span class="v">1.21</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>gttrf</code></th><td class="ok"><span class="v">1.5</span></td><td class="ok"><span class="v">1.48</span></td><td class="ok"><span class="v">1.43</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>gttrs</code></th><td class="ok"><span class="v">1.0</span></td><td class="ok"><span class="v">1.0</span></td><td class="ok"><span class="v">1.0</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>pbtrfL</code></th><td class="ok"><span class="v">1.07</span></td><td class="ok"><span class="v">1.13</span></td><td class="ok"><span class="v">1.06</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>pbtrfU</code></th><td class="b3"><span class="v">0.922</span><span class="n">n=128</span></td><td class="ok"><span class="v">1.02</span></td><td class="b3"><span class="v">0.933</span><span class="n">n=128</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>potrf</code></th><td class="ok"><span class="v">1.01</span></td><td class="b2"><span class="v">0.951</span><span class="n">n=2100</span></td><td class="b2"><span class="v">0.959</span><span class="n">n=8</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>potrfU</code></th><td class="b2"><span class="v">0.969</span><span class="n">n=1000</span></td><td class="b3"><span class="v">0.934</span><span class="n">n=1000</span></td><td class="b3"><span class="v">0.94</span><span class="n">n=1000</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>potri</code></th><td class="b2"><span class="v">0.957</span><span class="n">n=2048</span></td><td class="ok"><span class="v">1.03</span></td><td class="ok"><span class="v">1.03</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>potrsL</code></th><td class="ok"><span class="v">1.0</span></td><td class="ok"><span class="v">1.03</span></td><td class="ok"><span class="v">1.03</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>potrsU</code></th><td class="b2"><span class="v">0.959</span><span class="n">n=2048</span></td><td class="ok"><span class="v">1.02</span></td><td class="ok"><span class="v">1.01</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>pptrfL</code></th><td class="ok"><span class="v">1.16</span></td><td class="ok"><span class="v">1.12</span></td><td class="ok"><span class="v">1.15</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>pptrfU</code></th><td class="ok"><span class="v">2.03</span></td><td class="ok"><span class="v">1.81</span></td><td class="ok"><span class="v">2.09</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>pstrf</code></th><td class="b2"><span class="v">0.979</span><span class="n">n=1000</span></td><td class="ok"><span class="v">1.0</span></td><td class="ok"><span class="v">1.06</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>pstrfU</code></th><td class="b3"><span class="v">0.937</span><span class="n">n=2100</span></td><td class="b3"><span class="v">0.924</span><span class="n">n=50</span></td><td class="b2"><span class="v">0.966</span><span class="n">n=50</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>ptsv</code></th><td class="ok"><span class="v">1.05</span></td><td class="ok"><span class="v">1.05</span></td><td class="ok"><span class="v">1.07</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>pttrf</code></th><td class="ok"><span class="v">1.11</span></td><td class="ok"><span class="v">1.12</span></td><td class="ok"><span class="v">1.09</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>pttrs</code></th><td class="b2"><span class="v">0.975</span><span class="n">n=256</span></td><td class="b2"><span class="v">0.956</span><span class="n">n=256</span></td><td class="b2"><span class="v">0.988</span><span class="n">n=256</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>syev</code></th><td class="ok"><span class="v">1.13</span></td><td class="ok"><span class="v">1.09</span></td><td class="ok"><span class="v">1.09</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>syevN</code></th><td class="ok"><span class="v">1.02</span></td><td class="b2"><span class="v">0.972</span><span class="n">n=1024</span></td><td class="ok"><span class="v">1.08</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>sytrf</code></th><td class="ok"><span class="v">1.08</span></td><td class="b2"><span class="v">0.957</span><span class="n">n=2048</span></td><td class="ok"><span class="v">1.02</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>sytri</code></th><td class="b1"><span class="v">0.992</span><span class="n">n=2048</span></td><td class="b3"><span class="v">0.871</span><span class="n">n=1024</span></td><td class="b1"><span class="v">0.991</span><span class="n">n=1024</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>sytrs</code></th><td class="ok"><span class="v">1.43</span></td><td class="ok"><span class="v">1.39</span></td><td class="ok"><span class="v">1.29</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>trtri</code></th><td class="ok"><span class="v">1.04</span></td><td class="ok"><span class="v">1.02</span></td><td class="ok"><span class="v">1.07</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>trtrs</code></th><td class="b1"><span class="v">0.993</span><span class="n">n=50</span></td><td class="b3"><span class="v">0.9</span><span class="n">n=50</span></td><td class="b3"><span class="v">0.872</span><span class="n">n=100</span></td><td class="h"><span class="n">7aae0d6</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zgeqrf</code></th><td class="ok"><span class="v">1.13</span></td><td class="ok"><span class="v">1.02</span></td><td class="ok"><span class="v">1.07</span></td><td class="h"><span class="n">7b4942c</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zgesvd</code></th><td class="ok"><span class="v">1.01</span></td><td class="b3"><span class="v">0.93</span><span class="n">n=50</span></td><td class="ok"><span class="v">1.0</span></td><td class="h"><span class="n">7b4942c</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zgetrf</code></th><td class="ok"><span class="v">1.0</span></td><td class="b3"><span class="v">0.872</span><span class="n">n=50</span></td><td class="b3"><span class="v">0.896</span><span class="n">n=100</span></td><td class="h"><span class="n">7b4942c</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zheev</code></th><td class="ok"><span class="v">1.3</span></td><td class="ok"><span class="v">1.14</span></td><td class="ok"><span class="v">1.02</span></td><td class="h"><span class="n">7b4942c</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zheevN</code></th><td class="ok"><span class="v">1.11</span></td><td class="ok"><span class="v">1.06</span></td><td class="ok"><span class="v">1.13</span></td><td class="h"><span class="n">7b4942c</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zpotrf</code></th><td class="ok"><span class="v">1.04</span></td><td class="ok"><span class="v">1.06</span></td><td class="ok"><span class="v">1.01</span></td><td class="h"><span class="n">7b4942c</span></td><td class="n"><span class="n">?</span></td></tr>
<tr><th><code>zpotrfU</code></th><td class="ok"><span class="v">1.05</span></td><td class="ok"><span class="v">1.02</span></td><td class="b1"><span class="v">0.994</span><span class="n">n=32</span></td><td class="h"><span class="n">7b4942c</span></td><td class="n"><span class="n">?</span></td></tr>
</tbody></table></div>
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
| Cholesky (lower) | potrf | s/d/c/z | ✅ | 1.01 | **0.951** | **0.957** | 1.5 / 1 | 1.58 / 0.951 |
| Cholesky (upper) | potrf `uplo='U'` | s/d/c/z | ✅ | **0.969** | **0.934** | **0.943** | 1.42 / 1.09 | 1.37 / 0.934 |
| Cholesky solve | potrs | s/d/c/z | ✅ | **0.957** | 1.02 | 1.01 | 2.87 / 1.21 | 1.25 / 1.02 |
| Pivoted Cholesky | pstrf | s/d/c/z | ✅ | **0.938** | **0.924** | **0.966** | 1.31 / 0.924 | 1.26 / 0.958 |
| LU | getrf, gesv | s/d/c/z | ✅ | 1.01 | **0.94** | **0.953** | 1.26 / 1.01 | 1.44 / 0.94 |
| LU solve | getrs | s/d/c/z | ✅ | **0.96** | **0.832** | **0.824** | 1.23 / 1.03 | 1.09 / 0.832 |
| QR | geqrf, orgqr, ormqr | s/d/c/z | ✅ | **0.956** | **0.934** | **0.943** | 1.79 / 1.2 | 1.46 / 0.934 |
| Pivoted QR | geqp3 | s/d/c/z | ✅ | **0.899** | **0.955** | 0.997 | 1.32 / 1.02 | 1.16 / 0.955 |
| Bunch–Kaufman | sytrf, hetrf | s/d/c/z | ✅ | 1.07 | **0.957** | 1.01 | 1.38 / 1.05 | 1.34 / 0.957 |
| Bunch–Kaufman solve | sytrs, hetrs | s/d/c/z | ✅ | 1.43 | 1.39 | 1.29 | 1.96 / 1.54 | 2.26 / 1.39 |
| Triangular solve | trtrs | s/d/c/z | ✅ | **0.986** | **0.894** | **0.872** | 1.21 / 0.974 | 1.13 / 0.894 |
| Least-squares | gels | s/d/c/z | ✅ | 1.05 | 1.22 | 1.19 | 2.36 / 1.35 | 1.77 / 1.22 |
| SVD | gesvd, gesdd | s/d/c/z | ✅ | 1.04 | **0.958** | **0.967** | 1.31 / 1.05 | 1.24 / 0.958 |
| Symmetric eigen | syev, syevd, syevr | s/d/c/z | ✅ | 1.02 | **0.973** | 1.02 | 1.37 / 0.973 | 1.51 / 1.09 |

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
| Symmetric / Hermitian (vectors) | syev, syevd, syevr, heev, sytrd, hetrd, ormtr, *stedc\*, steqr\** | s/d/c/z | ✅ | 1.13 | 1.09 | 1.02 |
| Symmetric / Hermitian (values only) | syev, *sterf\** | s/d/c/z | ✅ | 1.02 | **0.973** | 1.08 |
| Sym-tridiagonal | stev, stegr, stebz, stein | s/d | ✅ | | | ⏳ |
| Generalized symmetric | sygvd, hegvd | s/d/c/z | ✅ | | | ⏳ |
| Nonsymmetric | geev, geevx, gebal, gehrd, hseqr, trevc, gebak | s/d/c/z | ✅ | | | ⏳ |
| Schur | gees | s/d/c/z | ✅ | | | ⏳ |
| Generalized nonsym (QZ) | ggev, gges, gghrd, hgeqz, tgevc | s/d/c/z | ✅ | | | ⏳ |
| Schur reordering | trexc, trsen | s/d/c/z | ✅ | | | ⏳ |
| Sylvester / Lyapunov | trsyl | s/d/c/z | ✅ | | | ⏳ |

*\* `stedc`, `steqr` and `sterf` are implemented but not forwarded. They are the internal building
blocks `syev`/`heev` are composed from, and PureBLAS exports no `stedc_64_`/`steqr_64_`/`sterf_64_`
symbol, so a program that calls one of them directly still reaches OpenBLAS. Everything else in these
two rows routes to PureBLAS. Checked against `src/cabi/`, 2026-08-20.*

## LAPACK — banded / tridiagonal / packed

| Op | Routines | Types | Routes | Zen3 | Zen4 | Zen5 | Zen4 vs OB geo/worst | Zen4 vs AOCL geo/worst |
|---|---|---|---|---|---|---|---|---|
| General banded LU | gbtrf, gbtrs | s/d/c/z | ✅ | 1.03 | **0.988** | **0.993** | 1.65 / 1.26 | 1.61 / 0.988 |
| General tridiagonal | gtsv, gttrf, gttrs | s/d/c/z | ✅ | 1 | 1 | 0.999 | 1.34 / 1 | 1.27 / 1.03 |
| SPD tridiagonal | pttrf, pttrs, ptsv | s/d/c/z | ✅ | **0.975** | **0.955** | **0.987** | 1.38 / 1.12 | 1.06 / 0.955 |
| Banded Cholesky | pbtrf, pbtrs | s/d/c/z | ✅ | **0.922** | 1.02 | **0.933** | 1.61 / 1.27 | 1.53 / 1.02 |
| Packed Cholesky | pptrf, pptrs | s/d/c/z | ✅ | 1.16 | 1.12 | 1.15 | 2.07 / 1.2 | 3.55 / 1.12 |

`pttrs`'s 0.99 vs AOCL is a shared dependency-chain bound, not a gap — see [Notes](notes.md).

## Free via composition

`exp`, `sqrt`, `log` and `^` of a matrix, `sylvester`/`lyap`, `pinv`, `nullspace`, `rank`, `cond` and
`factorize` are computed in Julia on top of the routed `eigen`/`schur`/`svd`/`\` kernels, so none of
them needs a LAPACK wrapper of its own.

## No OpenBLAS fallthrough

Every LAPACK symbol `LinearAlgebra` can `ccall` forwards to PureBLAS after `activate()`. That includes
the auxiliaries (`larf`/`larfg`/`lacpy`), the driver internals
(`gebrd`/`bdsqr`/`bdsdc`/`hseqr`/`trevc`/`gebak`/`sytrd`/`hetrd`/`orgtr`/`ormtr`), the combined and
expert drivers (`gesv`, `posv`, and `gesvx` with equilibration, iterative refinement and
condition/error bounds), the reordering routines (`trexc`/`trsen`/`tgsen`, real and complex — the real
path does the 2×2 conjugate-pair swap), `trrfs`, `syconv`, complex `bdsqr`, and the rank-deficient
generalized SVD (`ggsvd`, all four types). A ratchet test (`test/lbt_forward_tests.jl`) enumerates
every symbol the stdlib wraps and fails if the fallthrough count is anything but zero.

`cstev_` and `zstev_` are the only names left out of that count, and they are not real LAPACK symbols —
they appear only in commented-out stdlib lines and have no OpenBLAS export.

## Summary

Routing is complete. Every operation in the tables reaches PureBLAS after `activate()`, every one is
numerically LAPACK-accurate, and the ratchet test above holds the OpenBLAS fallthrough at zero.

Speed is usually ahead of both references, though not yet everywhere. On the Zen4 sweep the geomeans
run about 1.0–2.4× across BLAS and the dense factorizations, and 36 of the 83 measured rows clear
`max(OpenBLAS, AOCL)` at every size. The rest miss somewhere, mostly narrowly; [Notes](notes.md) says
which cells and why.

The eigensolvers (symmetric, Hermitian, nonsymmetric, generalized, Schur), Sylvester and Schur
reordering, `gesvx`, the generalized SVD and the remaining factorizations (indefinite, QL/RQ, RZ,
pivoted Cholesky, banded, tridiagonal, packed, rank-deficient least squares) are routed and
LAPACK-accurate, but they were written correctness-first — gating them is a scheduled follow-up.
