using Plots
using CSV
using DataFrames

csvfile = CSV.read("save_graph/visual_baseline5.csv")
x = csvfile[:1]
y = csvfile[:2]

smoothed = float(copy(y))

#= Smooth the array =#
alpha = 0.1
for i in 2:length(y)
    smoothed[i] = alpha * y[i] + (1. - alpha) * smoothed[i-1]
end
multiple = [y smoothed]

plot(x, multiple, title="Baseline Visual History A2C", linealpha = [0.5 1], label=["Raw Scores" "Smoothed Scores"])
#= plot(x, y, title="S Encoder A2C", label=["Scores"]) =#

savefig("save_graph/visual_baseline5.png")
