using OrthogonalSphericalShellGrids
using OrthogonalSphericalShellGrids.Oceananigans
using OrthogonalSphericalShellGrids: Face, Center
using OrthogonalSphericalShellGrids: get_cartesian_nodes_and_vertices
using Oceananigans.Fields: interpolate!
using Oceananigans.Grids: φnode
using Oceananigans.Operators

using GLMakie

# Here we assume that the tripolar grid is locally orthogonal
@inline function convert_to_latlong_frame(i, j, grid, uₒ, vₒ)

    φᶜᶠᵃ₊ = φnode(i, j+1, 1, grid, Center(), Face(), Center())
    φᶜᶠᵃ₋ = φnode(i,   j, 1, grid, Center(), Face(), Center())
    Δyᶜᶜᵃ = Δyᶜᶜᶜ(i,   j, 1, grid)

    ũ = deg2rad(φᶜᶠᵃ₊ - φᶜᶠᵃ₋) / Δyᶜᶜᵃ

    φᶠᶜᵃ₊ = φnode(i+1, j, 1, grid, Face(), Center(), Center())
    φᶠᶜᵃ₋ = φnode(i,   j, 1, grid, Face(), Center(), Center())
    Δxᶜᶜᵃ = Δxᶜᶜᶜ(i,   j, 1, grid)

    ṽ = - deg2rad(φᶠᶜᵃ₊ - φᶠᶜᵃ₋) / Δxᶜᶜᵃ

    𝒰 = sqrt(ũ^2 + ṽ^2)

    d₁ = ũ / 𝒰
    d₂ = ṽ / 𝒰

    return uₒ * d₁ - vₒ * d₂, uₒ * d₂ + vₒ * d₁
end

# Here we assume that the tripolar grid is locally orthogonal
@inline function convert_to_native_frame(i, j, grid, uₒ, vₒ)

    φᶜᶠᵃ₊ = φnode(i, j+1, 1, grid, Center(), Face(), Center())
    φᶜᶠᵃ₋ = φnode(i,   j, 1, grid, Center(), Face(), Center())
    Δyᶜᶜᵃ = Δyᶜᶜᶜ(i,   j, 1, grid)

    ũ = deg2rad(φᶜᶠᵃ₊ - φᶜᶠᵃ₋) / Δyᶜᶜᵃ

    φᶠᶜᵃ₊ = φnode(i+1, j, 1, grid, Face(), Center(), Center())
    φᶠᶜᵃ₋ = φnode(i,   j, 1, grid, Face(), Center(), Center())
    Δxᶜᶜᵃ = Δxᶜᶜᶜ(i,   j, 1, grid)

    ṽ = - deg2rad(φᶠᶜᵃ₊ - φᶠᶜᵃ₋) / Δxᶜᶜᵃ

    𝒰 = sqrt(ũ^2 + ṽ^2)

    d₁ = ũ / 𝒰
    d₂ = ṽ / 𝒰

    return uₒ * d₁ + vₒ * d₂, uₒ * d₂ - vₒ * d₁
end

# Generate a Tripolar grid with a 2 degree resolution and ``north'' singularities at 20 degrees latitude
grid = TripolarGrid(size = (180, 90, 1), north_poles_latitude = 35)

uLL = CenterField(grid)
vLL = CenterField(grid)

# We want a purely zonal velocity 
fill!(uLL, 1)

# Correct interpolation of uLL and vLL
@inline convert_x(i, j, k, grid, uₒ, vₒ) = convert_to_latlong_frame(i, j, grid, uₒ[i, j, k], vₒ[i, j, k])[1]
@inline convert_y(i, j, k, grid, uₒ, vₒ) = convert_to_latlong_frame(i, j, grid, uₒ[i, j, k], vₒ[i, j, k])[2]

uC = KernelFunctionOperation{Center, Center, Center}(convert_x, grid, uLL, vLL)
vC = KernelFunctionOperation{Center, Center, Center}(convert_y, grid, uLL, vLL)

uTR = compute!(Field(uC))
vTR = compute!(Field(vC))

# Correct interpolation of uLL and vLL
@inline convert_x_back(i, j, k, grid, uₒ, vₒ) = convert_to_native_frame(i, j, grid, uₒ[i, j, k], vₒ[i, j, k])[1]
@inline convert_y_back(i, j, k, grid, uₒ, vₒ) = convert_to_native_frame(i, j, grid, uₒ[i, j, k], vₒ[i, j, k])[2]

uB = KernelFunctionOperation{Center, Center, Center}(convert_x_back, grid, uTR, vTR)
vB = KernelFunctionOperation{Center, Center, Center}(convert_y_back, grid, uTR, vTR)

uB = compute!(Field(uB))
vB = compute!(Field(vB))