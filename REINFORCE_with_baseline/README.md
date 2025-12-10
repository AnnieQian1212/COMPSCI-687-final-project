/code contains all the code for REINFORCE with baseline algorithm
/Figure and /Result contain some of the experimental results from the algorithm running in different environments. Note those ones are not all the results. Some were saved locally and I did not want to clutter the repository.

In /code, .py files are named based on their functionalities. *_algo.py are files that contain the algorithm for said environments. *_nets.py are files that contain the neural network inputs for said environments. Those neural networks are adapted from policy network provided in HW2.

parallel.py is used for running _algo.py in parallel with different hyperparameters. 

example.py is an example file for me to remeber how to use the neural network, similar to the one provided in HW2.

To run the code, change `from <enviroment>_algo import *` to the named enviroment file from the directory. Then change the hyperparameter array in code file. Then `python parallel.py`