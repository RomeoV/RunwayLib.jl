using Pluto, Pkg, Test

@assert pkgversion(Pkg) >= v"1.13.0" "Pkg 1.13+ required for notebook source sections (see https://github.com/JuliaLang/Pkg.jl/pull/4225)"

const RUNWAYLIB_PKGDIR = get(ENV, "RUNWAYLIB_PKGDIR", pwd())
const NOTEBOOKS_DIR = joinpath(RUNWAYLIB_PKGDIR, "pluto_notebooks")

"""
    run_notebook_test(notebookfile)

Activate the notebook environment, dev RunwayLib into it, run all cells,
and return true if no cells failed.
"""
function run_notebook_test(notebookfile)
    Pluto.activate_notebook_environment(notebookfile)
    Pkg.develop(; name="RunwayLib", path=RUNWAYLIB_PKGDIR)

    session = Pluto.ServerSession()
    notebook = Pluto.SessionActions.open(session, notebookfile; run_async=false)

    # Check that no cells have errors
    failed_cells = filter(notebook.cells) do cell
        cell.errored
    end

    Pluto.SessionActions.shutdown(session, notebook; async=false)

    if !isempty(failed_cells)
        @error "Failed cells in $notebookfile:" failed_cells=map(c -> c.cell_id, failed_cells)
        return false
    end
    return true
end

# Find all notebooks
notebooks = filter(endswith(".jl"), readdir(NOTEBOOKS_DIR; join=true))

@testset "Pluto Notebooks" begin
    @testset "$(basename(nb))" for nb in notebooks
        @test run_notebook_test(nb)
    end
end
