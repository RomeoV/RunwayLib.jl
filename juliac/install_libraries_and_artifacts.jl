using PackageCompiler
ctx = PackageCompiler.create_pkg_context(".")
PackageCompiler.bundle_artifacts(ctx, ENV["COMPILED_DIR"]; include_lazy_artifacts=false)
stdlibs = PackageCompiler.gather_stdlibs_project(ctx)
PackageCompiler.bundle_julia_libraries(ENV["COMPILED_DIR"], stdlibs)
for (root, _, files) in walkdir(ENV["COMPILED_DIR"]), file in files
  (contains(file, "libjulia-codegen") || contains(file, "libLLVM")) && rm(joinpath(root, file); force=true)
end
