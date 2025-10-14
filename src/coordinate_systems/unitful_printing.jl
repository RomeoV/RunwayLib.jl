function Base.show(
    io::IO, ::MIME"text/plain", v::Union{WorldPoint{T},ProjectionPoint{T},CameraPoint{T}}
) where {T<:Unitful.Quantity}
    u = unit(zero(T))
    numtype = Unitful.numtype(T)

    print(io, length(v), "-element WorldPoint{", numtype, "{", u, "}} with indices ", axes(v, 1), ":")

    # Print the elements
    for i in eachindex(v)
        print(io, "\n ", v[i])
    end
end
