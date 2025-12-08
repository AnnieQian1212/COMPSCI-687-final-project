from cartpole_nets import *

alpha_w = 0.1
alpha = 0.1
gamma = 0.99

policy_neurons_per_layer = (4, 2)
value_neurons_per_layer = (4, 2)
policy = CartPolicyNet(neurons_per_layer=policy_neurons_per_layer)
value = CartValueNet(neurons_per_layer=value_neurons_per_layer)


theta = policy.get_parameters()
theta_params = theta.size

w = value.get_parameters()
w_params = w.size

print(f"Policy network has {theta_params} parameters.")
print(f"Value network has {w_params} parameters.")
print(f"\nPolicy Parameters: \n{theta}")
print(f"\nValue Parameters: \n{w}")

#Define
state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

action = policy.act(state)
print(f"\nSampled action from policy for state {state}: {action}")

G = 2
delta = G - value.predict(state)
grad_w = value.get_gradients(state)
print(f"\nValue function prediction for state {state}: {value.predict(state)}")
w = w + alpha_w * delta * grad_w
print(f"\nUpdated value parameters:\n{w}")

grad_ln_pi = policy.get_ln_pi_grad(state, action)
print(f"\nGradient of log-policy at state {state} and action {action}:\n{grad_ln_pi}")
theta = theta + alpha * gamma *delta * grad_ln_pi
print(f"\nUpdated policy parameters:\n{theta}")