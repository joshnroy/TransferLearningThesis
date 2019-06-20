#= using Pkg =#
#= Pkg.add("Plots") =#
#= Pkg.add("CSV") =#
#= Pkg.add("DataFrames") =#
#= Pkg.add("NPZ") =#
using Plots
using CSV
using DataFrames
using NPZ

function stretch_to_500k(input_data)
    stretched = Float64[]
    for x in input_data
        for i in range(1, stop=x)
            append!(stretched, x)
        end
    end
    return stretched
end

println("Loading Data")
vanillaCartpole = npzread("dqn_vanilla_history.npz")["episode_reward"]
visualCartpole = npzread("dqn_visual_history.npz")["episode_reward"]
stylegan = npzread("stylegan_dqn_training_history_500k_again.npz")["episode_reward"]
temporalVae = npzread("vae_dqn_training_history_500k_modified.npz")["episode_reward"]

smoothed_vanillaCartpole = float(copy(vanillaCartpole))
smoothed_visualCartpole = float(copy(visualCartpole))
smoothed_stylegan = float(copy(stylegan))
smoothed_temporalVae = float(copy(temporalVae))

#= Smooth the array =#
println("Smoothing Data")
alpha = 0.01
for i in 2:length(smoothed_vanillaCartpole)
    smoothed_vanillaCartpole[i] = alpha * vanillaCartpole[i] + (1. - alpha) * smoothed_vanillaCartpole[i-1]
end
for i in 2:length(smoothed_visualCartpole)
    smoothed_visualCartpole[i] = alpha * visualCartpole[i] + (1. - alpha) * smoothed_visualCartpole[i-1]
end
for i in 2:length(smoothed_stylegan)
    smoothed_stylegan[i] = alpha * stylegan[i] + (1. - alpha) * smoothed_stylegan[i-1]
end
for i in 2:length(smoothed_temporalVae)
    smoothed_temporalVae[i] = alpha * temporalVae[i] + (1. - alpha) * smoothed_temporalVae[i-1]
end
println("Clamping Data")
clamp!(smoothed_vanillaCartpole, 0, 500)
clamp!(smoothed_visualCartpole, 0, 500)
clamp!(smoothed_stylegan, 0, 500)
clamp!(smoothed_temporalVae, 0, 500)
clamp!(vanillaCartpole, 0, 500)
clamp!(visualCartpole, 0, 500)
clamp!(stylegan, 0, 500)
clamp!(temporalVae, 0, 500)

println("Stretching Data")
smoothed_vanillaCartpole = stretch_to_500k(smoothed_vanillaCartpole)
smoothed_visualCartpole = stretch_to_500k(smoothed_visualCartpole)
smoothed_stylegan = stretch_to_500k(smoothed_stylegan)
smoothed_temporalVae = stretch_to_500k(smoothed_temporalVae)

minLength = minimum([length(smoothed_vanillaCartpole) length(smoothed_visualCartpole) length(smoothed_stylegan) length(smoothed_temporalVae)])
println("minLength is ", minLength)

smoothed_vanillaCartpole = smoothed_vanillaCartpole[1:minLength]
smoothed_visualCartpole = smoothed_visualCartpole[1:minLength]
smoothed_stylegan = smoothed_stylegan[1:minLength]
smoothed_temporalVae = smoothed_temporalVae[1:minLength]

println("Plotting Data")
x_data = range(1, stop=minLength)
plot(x_data, [smoothed_vanillaCartpole smoothed_visualCartpole smoothed_stylegan smoothed_temporalVae], label=["Smoothed Vanilla Cartpole" "Smoothed Visual Cartpole" "Smoothed Stylegan" "Smoothed Temporal VAE"], xlabel="Number of Timesteps", ylabel="Reward", title="Reward vs Training Timestep", legend=:bottomright)
# plot(range(1, stop=length(smoothed_vanillaCartpole)), smoothed_vanillaCartpole, label="Smoothed Vanilla Cartpole")
# plot!(range(1, stop=length(smoothed_visualCartpole)), smoothed_visualCartpole, label="Smoothed Visual Cartpole")
# plot!(range(1, stop=length(smoothed_stylegan)), smoothed_stylegan, label="Smoothed Stylegan")
# plot!(range(1, stop=length(smoothed_temporalVae)), smoothed_temporalVae, label="Smoothed Temporal VAE")
# plot(range(1, stop=length(smoothed_vanillaCartpole)), [smoothed_vanillaCartpole vanillaCartpole], linealpha=[1 0.5], label=["Smoothed Vanilla Cartpole" "Vanilla Cartpole"])
# plot!(range(1, stop=length(smoothed_visualCartpole)), [smoothed_visualCartpole visualCartpole], linealpha=[1 0.5], label=["Smoothed Visual Cartpole" "Visual Cartpole"])
# plot!(range(1, stop=length(smoothed_stylegan)), [smoothed_stylegan stylegan], linealpha=[1 0.5], label=["Smoothed Stylegan" "Stylegan"])
# plot!(range(1, stop=length(smoothed_temporalVae)), [smoothed_temporalVae temporalVae], linealpha=[1 0.5], label=["Smoothed Temporal VAE" "Temporal VAE"])
# xlabel!("Number of Episodes")
# ylabel!("Reward")
# title!("DQN Reward vs Training Episode")

savefig("losses.png")