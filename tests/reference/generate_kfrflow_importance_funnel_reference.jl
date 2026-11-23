using Pkg

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(; temp=true)
Pkg.develop(path=joinpath(REPOSITORY_ROOT, "references", "KFRFlow.jl"))

using DelimitedFiles
using KFRFlow
using Random

const DIMENSION = 5
const NUM_PARTICLES = 20
const NUM_STEPS = 100
const DELTA_T = 0.01
const NUGGET = 1e-2

rng = MersenneTwister(0)
initial_particles = randn(rng, DIMENSION, NUM_PARTICLES)

function funnel_log_ratio(x)
    x1 = x[1]
    remaining_sq = sum(abs2, x[2:end])

    # Match the unnormalized funnel density used by the Julia experiment.
    log_target = -x1^2 / 18 - 0.5 * exp(-x1) * remaining_sq
    log_reference = -0.5 * sum(abs2, x)
    return log_target - log_reference
end

history, times = KFRFlowI(
    initial_particles,
    funnel_log_ratio;
    dt=DELTA_T,
    savehistory=true,
    pKernels=1.0,
    nugget=NUGGET,
)

@assert length(times) == NUM_STEPS + 1
@assert size(history) == (DIMENSION, NUM_PARTICLES, NUM_STEPS + 1)

reference = Matrix{Float64}(undef, length(times), 1 + DIMENSION * NUM_PARTICLES)
reference[:, 1] = times

for step in eachindex(times)
    reference[step, 2:end] = vec(history[:, :, step])
end

output_directory = joinpath(REPOSITORY_ROOT, "tests", "data")
mkpath(output_directory)
output_path = joinpath(output_directory, "kfrflow_importance_funnel_julia.csv")
writedlm(output_path, reference, ',')

println("Saved Julia KFRFlow-I funnel reference history to $output_path")
