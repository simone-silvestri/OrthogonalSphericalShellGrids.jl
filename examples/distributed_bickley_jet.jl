using Oceananigans
using Oceananigans.Units
using Printf
using OrthogonalSphericalShellGrids
using Oceananigans.Utils: get_cartesian_nodes_and_vertices

Nx = 360
Ny = 240
Nb = 40

first_pole_longitude = λ¹ₚ = 45
north_poles_latitude = φₚ  = 25

λ²ₚ = λ¹ₚ + 180

# Divide the y-direction in 4 CPU ranks
arch = CPU() # Distributed(CPU(), partition = Partition(1, 1))

# Build a tripolar grid with singularities at
# (0, -90), (45, 25), (225, 25)
underlying_grid = TripolarGrid(arch; size = (Nx, Ny, 1), 
                               halo = (5, 5, 5), 
                               first_pole_longitude,
                               north_poles_latitude)

# We need a bottom height field that ``masks'' the singularities
bottom_height(λ, φ) = ((abs(λ - λ¹ₚ) < 5) & (abs(φₚ - φ) < 5)) |
                      ((abs(λ - λ²ₚ) < 5) & (abs(φₚ - φ) < 5)) | (φ < -78) ? 1 : 0

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

Ψ(y) = - tanh(y) * 10
U(y) = sech(y)^2 

# A sinusoidal tracer
C(y, L) = sin(2π * y / L)

# Slightly off-center vortical perturbations
ψ̃(x, y, ℓ, k) = exp(-(y + ℓ/10)^2 / 2ℓ^2) * cos(k * x) * cos(k * y)

# Vortical velocity fields (ũ, ṽ) = (-∂_y, +∂_x) ψ̃
ũ(x, y, ℓ, k) = + ψ̃(x, y, ℓ, k) * (k * tan(k * y) + y / ℓ^2) 
ṽ(x, y, ℓ, k) = - ψ̃(x, y, ℓ, k) * k * tan(k * x) 

free_surface = SplitExplicitFreeSurface(grid; substeps = 30)

@info "Building a model..."; start=time_ns()

tracer_advection = Oceananigans.Advection.TracerAdvection(WENO(; order = 5), WENO(; order = 5), Centered())
momentum_advection = WENOVectorInvariant(vorticity_order = 5)

model = HydrostaticFreeSurfaceModel(; grid, free_surface,
                                      momentum_advection,
                                      tracer_advection,
                                      buoyancy = nothing,
                                      tracers = :c)

ζ = Oceananigans.Models.HydrostaticFreeSurfaceModels.VerticalVorticityField(model)    

# Parameters
ϵ = 0.1 # perturbation magnitude
ℓ = 0.5 # Gaussian width
k = 2.5 # Sinusoidal wavenumber

dr(x) = deg2rad(x)

# Total initial conditions
uᵢ(x, y, z) = U(dr(y)*8) + ϵ * ũ(dr(x)*2, dr(y)*8, ℓ, k)
vᵢ(x, y, z) = ϵ * ṽ(dr(x)*2, dr(y)*4, ℓ, k)
cᵢ(x, y, z) = C(dr(y)*8, 167.0)

set!(model, u=uᵢ, v=vᵢ, c=cᵢ)

Δt = 1minutes

wizard = TimeStepWizard(cfl=0.3, max_change=1.1, max_Δt=3hours)

simulation = Simulation(model, Δt=Δt, stop_time=500days)

rank = arch.local_rank

simulation.output_writers[:surface_tracer] = JLD2OutputWriter(model, merge(model.velocities, model.tracers, (; ζ)),
                                                              filename = "tripolar_bickley_$(rank).jld2", 
                                                              schedule = TimeInterval(1day),
                                                              overwrite_existing = true)

progress(sim) = @info @sprintf("rank %d, %s with %s, velocity: %.2e %.2e", rank, prettytime(time(sim)), prettytime(sim.Δt), maximum(sim.model.velocities.u), maximum(sim.model.velocities.v)) 

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))
simulation.callbacks[:wizard]   = Callback(wizard,   IterationInterval(10))

run!(simulation)
