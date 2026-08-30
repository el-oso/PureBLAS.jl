using Documenter, DocumenterVitepress, PureBLAS

makedocs(;
    sitename = "PureBLAS.jl",
    authors = "el_oso",
    modules = [PureBLAS],
    warnonly = true,
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/el-oso/PureBLAS.jl",
        devbranch = "master",
        devurl = "dev",
    ),
    draft = false,
    source = "src",
    build = "build",
    # GROUPED ON PURPOSE. DocumenterVitepress puts every TOP-LEVEL entry in the navbar, and ten
    # flat entries with names this long do not fit: the menu ran over the site title (rendering
    # "DesignPureBLAS.jl") on a narrow window and off the right edge on a wide one. A nested
    # `name => [...]` becomes a navbar dropdown instead (vitepress_config.jl `pagelist2str`),
    # so grouping keeps the bar to five short items. The sidebar still lists every page, which
    # is what the flat navbar was duplicating.
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md",
        "Internals" => [
            "Design" => "design.md",
            "SIMD & Hardware Adaptation" => "simd.md",
        ],
        "Tuning" => [
            "Tuning Constants" => "tuning.md",
            "Knob Registry" => "knobs.md",
        ],
        "Performance" => [
            "Benchmarks" => "performance.md",
            "LAPACK/BLAS Coverage" => "coverage.md",
            "Methodology & Provenance" => "methodology.md",
            "Performance Notes" => "notes.md",
        ],
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/el-oso/PureBLAS.jl",
    devbranch = "master",
    push_preview = true,
)
