#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

class SigmoidActivationFunction:
    def __call__(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        sig = self.__call__(x)
        return sig * (1 - sig)

class IdentityActivationFunction:
    def __call__(self, x: float) -> float:
        return x

    def derivative(self, x):
        return 1

class ConnectedLayer:
    def __init__(self, input_size, output_size, activation_func, weight_init_method, include_bias=True):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_func = SigmoidActivationFunction() if activation_func == 'sigmoid' else IdentityActivationFunction()
        self.weights = self.initialize_weights(weight_init_method, include_bias)

    def initialize_weights(self, method, include_bias):
        shape = (self.input_size + 1, self.output_size + 1) if include_bias else (self.input_size + 1, self.output_size)
        if method == 'zeroes':
            return np.zeros(shape, dtype=np.float64)
        elif method == 'random':
            return np.random.standard_normal(shape)
        else:
            raise ValueError("Unknown weight initialization method")

    def __str__(self) -> str:
        return str(self.weights)
    
    def compute_output(self, x):
        return self.activation_func(np.dot(x, self.weights))
    
    def backward_pass(self, zs, gradients):
        delta = np.dot(gradients[-1], self.weights.T) * self.activation_func.derivative(zs)
        return delta
    
    def update_weights(self, learning_rate, zs, gradients):
        gradient = np.dot(zs.T, gradients)
        self.weights -= learning_rate * gradient
        return gradient

class NeuralNetworkModel:
    def __init__(self, layers):
        self.layers = layers

    def forward_pass(self, x): 
        zs = [np.atleast_2d(np.append(1, x))]
        for layer in self.layers:
            out = layer.compute_output(zs[-1])
            zs.append(out)
        return float(zs[-1]), zs

    def backward_pass(self, zs, target, learning_rate=0.1, display=False):
        gradients = [zs[-1] - target]
        for i in range(len(zs) - 2, 0, -1):
            delta = self.layers[i].backward_pass(zs[i], gradients)
            gradients.append(delta)
        gradients.reverse()

        for i, layer in enumerate(self.layers):
            grad = layer.update_weights(learning_rate, zs[i], gradients[i])
            if display: print(f"Gradient of Layer {i + 1}: \n{grad}")

