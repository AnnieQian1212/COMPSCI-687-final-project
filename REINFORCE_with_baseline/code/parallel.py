a = [0.1,0.01,0.001]
a_w = [0.1,0.01,0.001]
gamma = [0.99,0.97,0.95]
policy_neurons_per_layer = [(32,16),(64,32),(32,32),(16,32)]
value_neurons_per_layer = [(32,16),(64,32),(32,32),(16,32)]


for alpha in a:
    for alpha_w in a_w:
        for g in gamma:
            for p_neurons in policy_neurons_per_layer:
                for v_neurons in value_neurons_per_layer: