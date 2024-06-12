using Oceananigans.Grids: halo_size, with_halo
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitState, SplitExplicitFreeSurface, calculate_column_height! 
using Oceananigans.Models.HydrostaticFreeSurfaceModels: augmented_kernel_offsets, augmented_kernel_size

import Oceananigans.Models.HydrostaticFreeSurfaceModels: materialize_free_surface, SplitExplicitAuxiliaryFields

# TODO: make sure this works on Distributed architectures, for the 
# moment the only distribution we allow with the tripolar grid is 
# a slab decomposition in the y-direction.
function SplitExplicitAuxiliaryFields(grid::TRG)

    Gᵁ = Field((Face,   Center, Nothing), grid)
    Gⱽ = Field((Center, Face,   Nothing), grid)

    bcs_fc = positive_zipper_boundary(Gᵁ, grid)
    bcs_cf = positive_zipper_boundary(Gⱽ, grid)
    
    # Hᶠᶜ and Hᶜᶠ do not follow the TripolarGrid convention that: fields on faces
    # need the sign to switch at the north halos. For this reason, we need to 
    # provide the boundary conditions manually.
    Hᶠᶜ = Field((Face,   Center, Nothing), grid; boundary_conditions = bcs_fc)
    Hᶜᶠ = Field((Center, Face,   Nothing), grid; boundary_conditions = bcs_cf)

    calculate_column_height!(Hᶠᶜ, (Face, Center, Center))
    calculate_column_height!(Hᶜᶠ, (Center, Face, Center))

    fill_halo_regions!((Hᶠᶜ, Hᶜᶠ))

    # In a non-parallel grid we calculate only the interior,
    # otherwise, the rules remain the same
    kernel_size    = tripolar_augmented_kernel_size(grid)
    kernel_offsets = tripolar_augmented_kernel_offsets(grid)

    kernel_parameters = KernelParameters(kernel_size, kernel_offsets)
    
    return SplitExplicitAuxiliaryFields(Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ, kernel_parameters)
end

positive_zipper_boundary(default_field, ::TRG) = 
        FieldBoundaryConditions(
            top    = nothing,
            bottom = nothing,
            west   = default_field.boundary_conditions.west,
            east   = default_field.boundary_conditions.east,         
            south  = default_field.boundary_conditions.south,
            north  = ZipperBoundaryCondition()
        )

function positive_zipper_boundary(default_field, grid::DTRG)  
        arch = architecture(grid)
        workers = ranks(arch.partition)

        if arch.local_rank == workers[2] - 1
                return  FieldBoundaryConditions(
                                top    = nothing,
                                bottom = nothing,
                                west   = default_field.boundary_conditions.west,
                                east   = default_field.boundary_conditions.east,         
                                south  = default_field.boundary_conditions.south,
                                north  = ZipperBoundaryCondition()
                        )
        else
                return default_field.boundary_conditions
        end
end

@inline tripolar_augmented_kernel_size(grid::TRG)    = (grid.Nx, grid.Ny + grid.Hy - 1)
@inline tripolar_augmented_kernel_offsets(grid::TRG) = (0, 0)
        
# In case of a distributed architecture, nothing changes!
@inline tripolar_augmented_kernel_size(grid::DTRG)    = augmented_kernel_size(grid)
@inline tripolar_augmented_kernel_offsets(grid::DTRG) = augmented_kernel_offsets(grid)

# We play the same trick as in the Distributed implementation and we extend the halos for
# a split explicit barotropic solver on a tripolar grid. Only on the North boundary though!
@inline tripolar_split_explicit_halos(old_halos, step_halo) = old_halos[1], max(step_halo, old_halos[2]), old_halos[3]

# Internal function for HydrostaticFreeSurfaceModel
function materialize_free_surface(free_surface::SplitExplicitFreeSurface, velocities, grid::TRG)

        settings  = free_surface.settings 

        old_halos  = halo_size(grid)
        Nsubsteps  = length(settings.substepping.averaging_weights)

        extended_halos = tripolar_split_explicit_halos(old_halos, Nsubsteps+1)         
        extended_grid  = with_halo(extended_halos, grid)

        Nze = size(extended_grid, 3)
        η = ZFaceField(extended_grid, indices = (:, :, Nze+1))

        return SplitExplicitFreeSurface(η,
                                        SplitExplicitState(extended_grid, settings.timestepper),
                                        SplitExplicitAuxiliaryFields(extended_grid),
                                        free_surface.gravitational_acceleration,
                                        free_surface.settings)
end
