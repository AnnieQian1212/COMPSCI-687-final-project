from frozen_nets import *
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os





def algorithm(alpha, alpha_w, gamma, policy_neurons_per_layer, value_neurons_per_layer):
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)       

    
        
    episodes = 5000
    T = 500

    

    policy = FrozenPolicyNet(neurons_per_layer=policy_neurons_per_layer)
    value = FrozenValueNet(neurons_per_layer=value_neurons_per_layer)

    results = np.zeros(episodes)

    os.makedirs('./Result', exist_ok=True)
    with open(f'./Result/Lake_runs_results_a{alpha}_aw{alpha_w}_g{gamma}_p{policy_neurons_per_layer}_v{value_neurons_per_layer}.txt', 'w') as f:
        f.write('')

    for ep in range(episodes):
        if ep % 100 == 0:
            print(f"Starting Episode {ep}/{episodes}")

        # Get episode data
        states, actions, rewards = [], [], []
        state, info = env.reset()
        
        for t in range(T):
            action = policy.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            
            if terminated or truncated:
                break

        episode_length = len(rewards)
        results[ep] = sum(rewards)
        
        
        for t in range(episode_length):
            G = 0
            for k in range(episode_length - t):
                G += (gamma**k) * rewards[t+k]
            
            
            delta = G - value.predict(states[t])
            
            grad_w = value.get_gradients(states[t])
            w = value.get_parameters()
            w = w + alpha_w * delta * grad_w
            value.set_parameters(w)

            
            grad_ln_pi = policy.get_ln_pi_grad(states[t], actions[t])
            theta = policy.get_parameters()
            theta = theta + alpha * (gamma ** t) * delta * grad_ln_pi
            policy.set_parameters(theta)
        
        if ep % 100 == 0:
            avg_reward = np.mean(results[max(0, ep-100):ep+1])
            print(f"  Episode length: {episode_length}, Avg100 reward: {avg_reward:.2f}")
        
        with open(f'./Result/Lake_runs_results_a{alpha}_aw{alpha_w}_g{gamma}_p{policy_neurons_per_layer}_v{value_neurons_per_layer}.txt', 'a') as f:
            f.write(f'Episode {ep}, Reward: {results[ep]}\n')

    # Plotting
    window = 100
    smoothed = np.convolve(results, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(13, 7))
    plt.plot(results, alpha=0.3, label='Raw', color='lightblue')
    plt.plot(range(window-1, len(results)), smoothed, label=f'{window}-episode average', 
            linewidth=2, color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('REINFORCE with Baseline on FrozenLake-v1')
    plt.suptitle(f'a: {alpha}, a_w: {alpha_w}, gamma: {gamma} \n Policy neurons: {policy_neurons_per_layer}, Value neurons: {value_neurons_per_layer}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'./Result/Lake_a{alpha}_aw{alpha_w}_g{gamma}_p{policy_neurons_per_layer}_v{value_neurons_per_layer}.png', dpi=150)
    plt.show()

    print(f"\nFinal 100-episode average: {np.mean(results[-100:]):.2f}")

# algorithm(0.01,0.01,0.99,(4,),(4,4))