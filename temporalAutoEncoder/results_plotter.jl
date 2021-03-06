#= using Pkg =#
#= Pkg.add("Plots") =#
#= Pkg.add("CSV") =#
#= Pkg.add("DataFrames") =#
using Plots
using CSV
using DataFrames

csvfile = CSV.read("save_graph/velocity_darla2.csv")
x = csvfile[:1]
y = csvfile[:2]
actor_reward = csvfile[:3]
critic_reward = csvfile[:4]

smoothed = float(copy(y))
smoothed_actor_reward = float(copy(actor_reward))
smoothed_critic_reward = float(copy(critic_reward))

#= Smooth the array =#
alpha = 0.1
for i in 2:length(actor_reward)
    #= smoothed[i] = alpha * y[i] + (1. - alpha) * smoothed[i-1] =#
    smoothed_actor_reward[i] = alpha * actor_reward[i] + (1. - alpha) * smoothed_actor_reward[i-1]
    smoothed_critic_reward[i] = alpha * critic_reward[i] + (1. - alpha) * smoothed_critic_reward[i-1]
end
clamp!(smoothed_actor_reward, 0, 500)
clamp!(smoothed_critic_reward, 0, 500)
#= multiple = [y smoothed] =#
#= rewards = [actor_reward smoothed_actor_reward critic_reward smoothed_critic_reward] =#
rewards = [smoothed_actor_reward * 10 smoothed_critic_reward]
#= rewards = [smoothed_actor_reward * 10 smoothed_critic_reward] =#
#= rewards = [smoothed_actor_reward] =#

#= plot(x, multiple, title="DARLA Visual History A2C", linealpha = [0.5 1], label=["Raw Scores" "Smoothed Scores"]) =#
plot(x, rewards, title="darla Visual History A2C (loss)", linealpha = [1 1], label=["Smoothed Actor Loss" "Smoothed Critic Loss"])
#= plot(x, rewards, title="DARLA Visual History A2C (loss)", linealpha = [1], label=["Smoothed Actor Loss"]) =#

savefig("save_graph/velocity_darla2_rewards.png")
