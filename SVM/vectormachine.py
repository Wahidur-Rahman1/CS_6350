#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from math import exp
import numpy as np
import scipy.optimize

class LinearSVM:
    def __init__(self, X, y, learning_rate_schedule, regularization_coefficient, training_epochs=10):
        self.weights = np.zeros(X.shape[1] + 1)  # Initialize weights including bias term
        self.train(X, y, learning_rate_schedule, regularization_coefficient, training_epochs)

    def train(self, X, y, learning_rate_schedule, regularization_coefficient, training_epochs=10):
        X = np.insert(X, 0, 1, axis=1)  # Add bias term to data

        for epoch in range(training_epochs):
            learning_rate = learning_rate_schedule(epoch)
            indices = np.arange(len(X))
            np.random.shuffle(indices)

            for i in indices:
                if y[i] * np.dot(self.weights, X[i]) <= 1:
                    self.weights[0] += learning_rate * regularization_coefficient * y[i]  # Update bias separately
                    self.weights[1:] -= learning_rate * self.weights[1:]  # Regularization term
                    self.weights[1:] += learning_rate * regularization_coefficient * y[i] * X[i][1:]  # Update weights
                else:
                    self.weights[1:] *= (1 - learning_rate)  # Regularization term for weights without bias

    def make_prediction(self, X) -> np.ndarray:
        X = np.insert(X, 0, 1, axis=1)  # Add bias term to data
        return np.sign(np.dot(X, self.weights))  # Predict using sign of dot product

def gaussian_kernel(x, y, gamma):
    return exp(-(np.linalg.norm(x-y, ord=2)**2) / gamma)

class KernelizedSVM:
    def __init__(self, X, y, regularization_coefficient, kernel='dot', gamma=None):
        self.weights = np.zeros_like(X[0])
        self.bias = 0.0
        self.support_vectors = []
        self.train(X, y, regularization_coefficient, kernel, gamma)

    def train(self, X, y, regularization_coefficient, kernel='dot', gamma=None):
        def objective_function(lagrange_multipliers, X, y):
            label_matrix = y * np.ones((len(y), len(y)))
            multiplier_matrix = lagrange_multipliers * np.ones((len(lagrange_multipliers), len(lagrange_multipliers)))
            if kernel == 'dot':
                kernel_values = X @ X.T
            elif kernel == 'gaussian':
                kernel_values = np.array([[gaussian_kernel(xi, xj, gamma) for xj in X] for xi in X])
            total_values = label_matrix * label_matrix.T * multiplier_matrix * multiplier_matrix.T * kernel_values
            return 0.5 * np.sum(total_values) - np.sum(lagrange_multipliers)

        constraints = [
            {'type': 'ineq', 'fun': lambda a: a},
            {'type': 'ineq', 'fun': lambda a: regularization_coefficient - a},
            {'type': 'eq', 'fun': lambda a: np.dot(a, y)}
        ]

        res = scipy.optimize.minimize(objective_function, x0=np.zeros(shape=(len(X),)), args=(X, y), method='SLSQP', constraints=constraints, tol=0.01)
        self.weights = np.sum(res.x[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)

        if kernel == 'dot':
            self.bias = np.mean(y - np.dot(X, self.weights))
        elif kernel == 'gaussian':
            self.bias = np.mean([y[i] - gaussian_kernel(self.weights, X[i], gamma) for i in range(len(X))])

        self.support_vectors = X[res.x > 1e-10]

    def make_prediction(self, X, kernel='dot', gamma=None) -> np.ndarray:
        if kernel == 'dot':
            return np.sign(np.dot(X, self.weights) + self.bias)
        elif kernel == 'gaussian':
            return np.sign(np.array([np.sum(gaussian_kernel(xi, xj, gamma) * self.weights[j] for j, xj in enumerate(self.support_vectors)) + self.bias for xi in X]))

