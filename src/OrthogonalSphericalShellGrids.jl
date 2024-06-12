module OrthogonalSphericalShellGrids

# The only things we need!
export TripolarGrid, ZipperBoundaryCondition

using Printf
using Oceananigans
using Oceananigans: Face, Center
using Oceananigans.Grids: R_Earth

using Oceananigans.Fields: index_binary_search
using Oceananigans.Architectures: device, on_architecture
using JLD2
using Adapt 

using Oceananigans.Grids: halo_size, spherical_area_quadrilateral
using Oceananigans.Grids: lat_lon_to_cartesian
using Oceananigans.Utils: get_cartesian_nodes_and_vertices
using OffsetArrays
using Oceananigans.Operators

using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Grids: generate_coordinate

using Oceananigans.BoundaryConditions
using CUDA: @allowscalar

@inline convert_to_0_360(x) = ((x % 360) + 360) % 360

## Correcting Oceananigans
## TODO: remove after accepted in Oceananigans

using Oceananigans.Grids: topology
using Adapt: adapt
import Adapt: adapt_structure

function Adapt.adapt_structure(to, grid::OrthogonalSphericalShellGrid)
    TX, TY, TZ = topology(grid)

    return OrthogonalSphericalShellGrid{TX, TY, TZ}(nothing,
                                                    grid.Nx, grid.Ny, grid.Nz,
                                                    grid.Hx, grid.Hy, grid.Hz,
                                                    grid.Lz,
                                                    adapt(to, grid.λᶜᶜᵃ),
                                                    adapt(to, grid.λᶠᶜᵃ),
                                                    adapt(to, grid.λᶜᶠᵃ),
                                                    adapt(to, grid.λᶠᶠᵃ),
                                                    adapt(to, grid.φᶜᶜᵃ),
                                                    adapt(to, grid.φᶠᶜᵃ),
                                                    adapt(to, grid.φᶜᶠᵃ),
                                                    adapt(to, grid.φᶠᶠᵃ),
                                                    adapt(to, grid.zᵃᵃᶜ),
                                                    adapt(to, grid.zᵃᵃᶠ),
                                                    adapt(to, grid.Δxᶜᶜᵃ),
                                                    adapt(to, grid.Δxᶠᶜᵃ),
                                                    adapt(to, grid.Δxᶜᶠᵃ),
                                                    adapt(to, grid.Δxᶠᶠᵃ),
                                                    adapt(to, grid.Δyᶜᶜᵃ),
                                                    adapt(to, grid.Δyᶜᶠᵃ),
                                                    adapt(to, grid.Δyᶠᶜᵃ),
                                                    adapt(to, grid.Δyᶠᶠᵃ),
                                                    adapt(to, grid.Δzᵃᵃᶜ),
                                                    adapt(to, grid.Δzᵃᵃᶠ),
                                                    adapt(to, grid.Azᶜᶜᵃ),
                                                    adapt(to, grid.Azᶠᶜᵃ),
                                                    adapt(to, grid.Azᶜᶠᵃ),
                                                    adapt(to, grid.Azᶠᶠᵃ),
                                                    adapt(to, grid.radius),
                                                    adapt(to, grid.conformal_mapping))
end

include("grid_utils.jl")
include("zipper_boundary_condition.jl")
include("generate_tripolar_coordinates.jl")
include("tripolar_grid.jl")
include("grid_extensions.jl")
include("with_halo.jl")
include("distributed_tripolar_grid.jl")
include("split_explicit_free_surface.jl")

end
