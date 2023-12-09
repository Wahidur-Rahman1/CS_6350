#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

import NeuralNetwork as nn

print("Result")
net = nn.NeuralNetworkModel([
    nn.ConnectedLayer(input_size=4, output_size=5, activation_func='sigmoid', weight_init_method='random'), # input
    nn.ConnectedLayer(input_size=5, output_size=5, activation_func='sigmoid', weight_init_method='random'), # hidden
    nn.ConnectedLayer(input_size=5, output_size=1, activation_func='identity', weight_init_method='random', include_bias=False) # output
])

x = np.array([3.8481, 10.1539, -3.8561, -4.2228])
ystar = 0
y, activations = net.forward_pass(x)
net.backward_pass(activations, ystar, display=True)


# In[ ]:




