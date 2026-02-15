using Documenter
using DocumenterCitations
using DocumenterInterLinks
using RunwayLib

# Generate python_interface.md from template
template = read(joinpath(@__DIR__, "src", "python_interface_template.md"), String)
example = read(joinpath(@__DIR__, "..", "test", "python_interface", "smoke_test.py"), String)
output = replace(template, "{{PYTHON_EXAMPLE}}" => example)
write(joinpath(@__DIR__, "src", "python_interface.md"), output)

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:numeric
)

links = InterLinks(
    "Rotations" => (
        "https://juliageometry.github.io/Rotations.jl/dev/",
        "https://juliageometry.github.io/Rotations.jl/dev/objects.inv"
    ),
    "ProbabilisticParameterEstimators" => (
        "https://romeov.github.io/ProbabilisticParameterEstimators.jl/dev/",
        "https://romeov.github.io/ProbabilisticParameterEstimators.jl/dev/objects.inv",
    ),
    "NonlinearSolve" => (
        "https://docs.sciml.ai/NonlinearSolve/stable/",
        "https://docs.sciml.ai/NonlinearSolve/stable/objects.inv",
    ),
)

makedocs(
    sitename="RunwayLib",
    format=Documenter.HTML(
        assets=String["assets/citations.css"],
        size_threshold_ignore=["benchmarks.md"],
    ),
    modules=[RunwayLib],
    repo=Remotes.GitHub("RomeoV", "RunwayLib.jl"),
    pages=[
        "RunwayLib.jl: Fast Pose Estimation and Runtime Assurance for Runway Landings." => "index.md",
        "camera_model.md",
        "python_interface.md",
        "C_interface.md",
        "noise_models.md",
        "integrity_check.md",
        "caches.md",
        "performance_tips.md",
        "benchmarks.md",
        "uncertainty_predictions.md",
        "api_reference.md",
    ],
    plugins=[bib, links]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo="github.com/RomeoV/RunwayLib.jl.git",
    devbranch="master",
    push_preview=true
)
