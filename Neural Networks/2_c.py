#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from os import makedirs
import matplotlib.pyplot as plt
import NeuralNetwork as nn 

try: 
    makedirs("./out/")
except FileExistsError: 
    pass

def mse_loss(pred, target):
    return 0.5 * (pred - target) ** 2

def load_data(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        for line in f:
            terms = line.strip().split(",")
            terms_flt = list(map(np.float64, terms))
            x.append(terms_flt[:-1])
            y.append(terms_flt[-1])
    return np.array(x), np.array(y)

train_x, train_y = load_data('train.csv')
test_x, test_y = load_data('test.csv')

def train_and_test(num_epochs, net, train_x, train_y, test_x, test_y, lr_0=0.1, d=1):
    train_errors = []
    test_errors = []

    for e in range(num_epochs):
        train_losses = []
        for i in range(len(train_x)):
            y, activations = net.forward_pass(train_x[i])
            train_losses.append(mse_loss(y, train_y[i]))
            lr = lr_0 / (1 + (lr_0 / d) * e)
            net.backward_pass(activations, train_y[i], lr)

        train_error = np.mean(train_losses)
        train_errors.append(train_error)

        test_losses = [mse_loss(net.forward_pass(test_x[i])[0], test_y[i]) for i in range(len(test_x))]
        test_error = np.mean(test_losses)
        test_errors.append(test_error)

        print(f"Epoch {e+1} - Training Error: {train_error:>8f}, Testing Error: {test_error:>8f}")

    return train_errors, test_errors

def plot_errors(train_errors, test_errors, width):
    fig, ax = plt.subplots()
    ax.plot(train_errors, label='Training Error')
    ax.plot(test_errors, label='Testing Error')
    ax.set_title(f"Training and Testing Errors - Width {width}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    plt.savefig(f"./out/width_{width}.png")

widths = [5, 10, 25, 50, 100]
learning_rates = [0.1, 0.1, 0.05, 0.01, 0.003]
decay = [1, 1, 1, 1, 2]

for width, lr, d in zip(widths, learning_rates, decay):
    print(f"\nWidth = {width}:\n-------------------------------")
    net = nn.NeuralNetworkModel([
        nn.ConnectedLayer(input_size=4, output_size=width, activation_func='sigmoid', weight_init_method='zeroes'),
        nn.ConnectedLayer(input_size=width, output_size=width, activation_func='sigmoid', weight_init_method='zeroes'),
        nn.ConnectedLayer(input_size=width, output_size=1, activation_func='identity', weight_init_method='zeroes', include_bias=False)
    ])

    training_errors, testing_errors = train_and_test(50, net, train_x, train_y, test_x, test_y, lr_0=lr, d=d)
    plot_errors(training_errors, testing_errors, width)

plt.show()



# In[ ]:




