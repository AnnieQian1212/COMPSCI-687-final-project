import random
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import numpy as np
from taxi_algo import *
import random
import math
import matplotlib.pyplot as plt
import os






class Parallelize:
    def __init__(self, f):
        self.f = f

    def _run(self, args):
        return self.f(*args)

    def run(self, args):
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
            outputs = list(ex.map(self._run, args))
            # outputs = np.asarray(outputs, dtype=np.float32)
        return outputs
    



  
                    
if __name__ == '__main__':

    a = [0.00001,]
    a_w = [0.00001,]
    gamma = [0.95]
    policy_neurons_per_layer = [(32, 32)]
    value_neurons_per_layer = [(16, 16), (32, 16), (32, 32),(16,16,16)]

    # a = [0.00005]
    # a_w = [0.0001]
    # gamma = [0.99]
    # policy_neurons_per_layer = [(16,16)]
    # value_neurons_per_layer = [(16,16)]

    hyperpara = []
    parallel = Parallelize(algorithm)
    for alpha in a:
        for alpha_w in a_w:
            for g in gamma:
                for p_neurons in policy_neurons_per_layer:
                    for v_neurons in value_neurons_per_layer:
                        
                        hyperpara.append([alpha, alpha_w, g, p_neurons, v_neurons])
                        
                    
    random.shuffle(hyperpara)
    print(hyperpara)
    arguments_to_parallel_function = [ hyperpara[i]
                                                    for i in range(len(hyperpara))]

    outputs = parallel.run(arguments_to_parallel_function)