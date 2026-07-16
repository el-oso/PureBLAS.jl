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
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md",
        "Design" => "design.md",
        "SIMD & Hardware Adaptation" => "simd.md",
        "Performance" => "performance.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/el-oso/PureBLAS.jl",
    devbranch = "master",
    push_preview = true,
)
