using Unitful: Quantity, Units, Unitlike, dimension, unit

"""
    WithDims(q::Quantity)
    WithDims(u::Units)
Returns a subtype of `Unitful.Quantity` with the dimensions constrained to the
dimension of `q` or `u`.
Useful to build unitful interfaces that don't constrain the numeric type or the unit, just the dimension of a quantity.
Examples:
```jldoctest
julia> using Unitful, Unitful.DefaultSymbols; import Unitful.hr

julia> using RunwayLib: WithDims

julia> circumference_of_square(side::WithDims(m)) = 4*side;

julia> circumference_of_square((1//2)m)  # works
2//1 m

julia> circumference_of_square((1//2)km)  # also works
2//1 km

julia> # You can also constrain the return type. The numeric type is usually inferred automatically.

julia> kinetic_energy(mass::WithDims(kg), velocity::WithDims(m/s))::WithDims(J) = mass*velocity^2;

julia> kinetic_energy(1000kg, 100km/hr)
10000000 kg km^2 hr^-2
```
See also [`WithUnits`](@ref).
"""
WithDims(q::Quantity) = Quantity{T,dimension(q),U} where {T<:Real,U<:Unitlike}
WithDims(u::Units) = Quantity{T,dimension(u),U} where {T<:Real,U<:Unitlike}

"""
    WithUnits(q::Quantity)
    WithUnits(u::Units)
Returns a subtype of `Unitful.Quantity` with the dimensions and units constrained to the
dimension and units of `q` or `u`.
Useful to build unitful interfaces that don't constrain the unit, but not the numeric type of a quantity.
Examples:
```jldoctest
julia> using Unitful, Unitful.DefaultSymbols; import Unitful.hr

julia> using RunwayLib: WithUnits

julia> circumference_of_square(side::WithUnits(m)) = 4*side;

julia> circumference_of_square((1//2)m)  # works
2//1 m

julia> # circumference_of_square((1//2)km)  # doesn't work, constrained to exactly meters

julia> # You can also constrain the return type. The numeric type is usually inferred automatically.

julia> kinetic_energy(mass::WithUnits(kg), velocity::WithUnits(m/s))::WithUnits(J) = mass*velocity^2 |> x->uconvert(J, x);

julia> kinetic_energy(1000kg, uconvert(m/s, 100km/hr))
62500000//81 J
```
See also [`WithDims`](@ref).
"""
WithUnits(q::Quantity) = Quantity{T,dimension(q),unit(q)} where {T<:Real}
WithUnits(u::Units) = Quantity{T,dimension(u),typeof(u)} where {T<:Real}
