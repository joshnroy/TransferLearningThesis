using Plots
using CSV
using DataFrames
using NPZ

filename = "darla_log_temp.txt"
temporal_filename = "attention_log_temp.txt"

println("Loading Data")
data = []
open(filename) do f
    for line in eachline(f)
        num = parse(Float64, line)
        append!(data, num)
    end
end

temporal_data = []
open(temporal_filename) do f
    for line in eachline(f)
        num = parse(Float64, line)
        append!(temporal_data, num)
    end
end

minlength = min(length(data), length(temporal_data))
println("minlength is ", minlength)

data = data[1:minlength]
temporal_data = temporal_data[1:minlength]

coeff = 0.001

datasmoothed = zeros(minlength)
temporal_datasmoothed = zeros(minlength)

datasmoothed[1] = data[1]
temporal_datasmoothed[1] = temporal_data[1]
for i ∈ range(2, stop=minlength)
    datasmoothed[i] = coeff * data[i] + (1. - coeff) * datasmoothed[i-1]
    temporal_datasmoothed[i] = coeff * temporal_data[i] + (1. - coeff) * temporal_datasmoothed[i-1]
end

println("Plotting Data")
plot([datasmoothed data temporal_datasmoothed temporal_data], label=["DARLA Smoothed" "DARLA" "Attention Smoothed" "Attention"],
                                                                     linealpha=[1 0.4 1 0.4],
                                                                     xlabel="Number of Timesteps",
                                                                     ylabel="Reward",
                                                                     title="Reward vs Training Timestep",
                                                                     legend=:bottomright)

savefig("rewards.png")
