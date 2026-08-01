# Pool the BLAS rows of docs/src/coverage.md straight out of the plots.jl caches.
#
# WHY THIS EXISTS. The four BLAS rows are a geomean/worst over every op AND size AND box in a group.
# Reading those off the per-uarch `gen_table_*.md` by eye and combining them mentally is how a table
# ends up half-fresh and half-stale with nothing saying which. This derives them from the cache files
# that carry their own provenance header, so the doc can be regenerated from evidence.
#
# Usage:  awk -f bench/coverage_rows.awk bench/plots_data_<hostA>[_aocl].txt bench/plots_data_<hostB>...
#         Pass ALL boxes for one reference at once — the pooling is across the fleet, per the doc.
#
# Row membership mirrors docs/src/coverage.md exactly: the dense row is gemv/ger/symv/trmv/trsv in
# BOTH real and complex (the doc lists one row for types s/d/c/z), banded/packed is gbmv/sbmv/spmv
# and their complex analogues. An op not listed in either set is reported under UNCLASSIFIED rather
# than silently dropped — a new op must be placed deliberately, not vanish from the coverage numbers.
BEGIN {
    split("gemvN gemvT ger symv trmv trsv zgemvN zgemvT zgemvC zgeru zhemv ztrmv ztrsv", d, " ")
    for (i in d) dense[d[i]] = 1
    split("spmv gbmvN sbmv zhpmv zgbmvN zhbmv", b, " ")
    for (i in b) bandpack[b[i]] = 1
}
/^#/ { next }
{
    lvl = $1; op = $2
    if (lvl == "L1" || lvl == "CL1")      row = "BLAS-1"
    else if (lvl == "L3" || lvl == "CL3") row = "BLAS-3"
    else if (lvl == "L2" || lvl == "CL2") {
        if (op in dense)         row = "BLAS-2 dense"
        else if (op in bandpack) row = "BLAS-2 banded/packed"
        else                     row = "UNCLASSIFIED(" op ")"
    } else next

    k = split($3, S, ";")
    for (i = 1; i <= k; i++) {
        split(S[i], a, "=")
        m = split(a[2], v, ",")
        s = 0; for (j = 1; j <= m; j++) s += v[j]
        r = s / m                                  # mean over this cell's samples
        g[row] += log(r); c[row]++
        if (!(row in w) || r < w[row]) { w[row] = r; wn[row] = op "@" a[1] }
        if (r < 1.0) { nfail[row]++; fl[row] = fl[row] sprintf("%s@%s=%.3f ", op, a[1], r) }
    }
}
END {
    for (x in g)
        printf "%-22s geo=%.3f  worst=%.3f (%s)   cells=%d  below1.0=%d\n",
               x, exp(g[x] / c[x]), w[x], wn[x], c[x], nfail[x] + 0
    print ""
    for (x in fl) printf "  %s misses: %s\n", x, fl[x]
}
