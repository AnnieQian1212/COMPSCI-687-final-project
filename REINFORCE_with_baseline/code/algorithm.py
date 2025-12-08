from cartpole_nets import *
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os

env = gym.make("CartPole-v1")       
alpha_w = 0.01
alpha = 0.001
gamma = 0.97
episodes = 2000
T = 500
policy_neurons_per_layer = (64, 64)
value_neurons_per_layer = (64, 32)
policy = CartPolicyNet(neurons_per_layer=policy_neurons_per_layer)
value = CartValueNet(neurons_per_layer=value_neurons_per_layer)
# theta = policy.get_parameters()
# policy.set_parameters(np.random.randn(*theta.shape).astype(np.float32).seed )
# w = value.get_parameters()
# value.set_parameters(np.zeros_like(w))

results = np.zeros(episodes)

os.makedirs('./Result', exist_ok=True)
with open('./Result/runs_results.txt', 'w') as f:
    f.write('')

for ep in range(episodes):
    if (ep) % 100 == 0:
        print(f"Starting Episode {ep}/{episodes}")

    #Get episode data
    states, actions, rewards = [],[], []
    state, info = env.reset()
    t = 0
    while t < T:
        action = policy.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        # print(f"Episode {ep}, Step {t}, State: {state}, Action: {action}, Reward: {reward}")
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
        t += 1
        if terminated or truncated:
            break

    t =0
    this_episode_length = len(rewards)
    while t < this_episode_length:
        
        G = sum([ (gamma**k)*rewards[t+k] for k in range(this_episode_length - t) ])
        if(t == 0):
            results[ep] = G
            with open('./Result/runs_results.txt', 'a') as f:
                f.write(f'Episode {ep}, Reward: {results[ep]}\n')
        delta = G - value.predict(states[t])
        grad_w = value.get_gradients(states[t])
        w = value.get_parameters()
        w = w + alpha_w * delta * grad_w
        value.set_parameters(w)

        grad_ln_pi = policy.get_ln_pi_grad(states[t], actions[t])
        theta = policy.get_parameters()
        theta = theta + alpha * (gamma ** t) * delta * grad_ln_pi
        policy.set_parameters(theta)

        t += 1

    
    


plt.plot(results)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('REINFORCE with Baseline on CartPole-v1')
plt.suptitle(f'Alpha: {alpha}, Alpha_w: {alpha_w}, Gamma: {gamma} with Policy Neurons {policy_neurons_per_layer} and Value Neurons {value_neurons_per_layer}')

plt.savefig(f'./Figure/CartPole_v1_a{alpha}_aw{alpha_w}_g{gamma} \n policy{policy_neurons_per_layer}_value{value_neurons_per_layer}.png')
plt.show()