using JuliaC
builddir = isempty(ARGS) ? "build" : ARGS[1]

img = ImageRecipe(
    output_type="--output-lib",
    file="./loadrunwaylib.jl",
    project="..",
    trim_mode="safe",
    add_ccallables=true,
    verbose=true
)
link = LinkRecipe(
    image_recipe=img,
    outname=joinpath(builddir, "lib"),
    rpath=nothing
)

bun = BundleRecipe(
    link_recipe=link,
    output_dir=builddir
)

compile_products(img)
link_products(link)
bundle_products(bun)
