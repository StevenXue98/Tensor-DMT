using Pkg

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(; temp=true)
Pkg.develop(path=joinpath(REPOSITORY_ROOT, "references", "KFRFlow.jl"))

using DelimitedFiles
using KFRFlow
using LinearAlgebra
using Random

const DIMENSION = 2
const NUM_PARTICLES = 20
const NUM_STEPS = 100
const DELTA_T = 0.01
const BANDWIDTH = 45.0
const NUGGET = 1e-8

rng = MersenneTwister(0)
initial_particles = randn(rng, DIMENSION, NUM_PARTICLES)

log_ratio(x) = -8.0 * (norm(x) - 2.0)^2

history, times = KFRFlowEuler(
    initial_particles,
    log_ratio;
    dt=DELTA_T,
    savehistory=true,
    pKernels=1.0,
    k=KFRFlow.MaternKernel(),
    bw=BANDWIDTH,
    nugget=NUGGET,
)

@assert length(times) == NUM_STEPS + 1
@assert size(history) == (DIMENSION, NUM_PARTICLES, NUM_STEPS + 1)

# Julia stores particles as columns in a d x J matrix. vec therefore places
# each particle's d coordinates consecutively, matching a NumPy reshape to
# (J, d) with C ordering.
reference = Matrix{Float64}(undef, length(times), 1 + DIMENSION * NUM_PARTICLES)
reference[:, 1] = times

for step in eachindex(times)
    reference[step, 2:end] = vec(history[:, :, step])
end

output_directory = joinpath(REPOSITORY_ROOT, "tests", "data")
mkpath(output_directory)
output_path = joinpath(output_directory, "kfrflow_donut_julia.csv")
writedlm(output_path, reference, ',')

println("Saved Julia KFRFlow reference history to $output_path")
