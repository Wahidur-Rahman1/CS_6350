#!/usr/bin/env python
# coding: utf-8

# In[26]:


import vectormachine  
from os import makedirs
import numpy as np
np.random.seed(33)

try: makedirs("./out/")
except FileExistsError: None

def read_data(file_path):
    data_x, data_y = [], []
    with open(file_path, "r") as file:
        for line in file:
            terms = line.strip().split(",")
            terms_flt = list(map(float, terms))
            data_x.append(terms_flt[:-1])
            data_y.append(-1 if terms_flt[-1] == 0 else 1)
    return np.array(data_x), np.array(data_y)

train_x, train_y = read_data("train.csv")
test_x, test_y = read_data("test.csv")

# Linear SVM with first learning rate schedule
print("Linear SVM, a)")
Cs = [100/873, 500/873, 700/873]
for C in Cs:
    print(f"C = {C}")
    lnot, a = 1, 1
    learning_rate_schedule_a = lambda e : lnot / (1 + (lnot/a)*e)
    linear_svm_a = vectormachine.LinearSVM(train_x, train_y, learning_rate_schedule_a, C, 100)
    print(f"Learned weights: {linear_svm_a.weights[1:]}")
    print(f"Learned bias: {linear_svm_a.weights[0]}")
    print(f"Training accuracy: {np.mean(train_y == linear_svm_a.make_prediction(train_x))}")
    print(f"Testing accuracy: {np.mean(test_y == linear_svm_a.make_prediction(test_x))}")

# Linear SVM with second learning rate schedule
print("Linear SVM, b)")
for C in Cs:
    print(f"C = {C}")
    lnot = 1
    learning_rate_schedule_b = lambda e : lnot / (1 + e)
    linear_svm_b = vectormachine.LinearSVM(train_x, train_y, learning_rate_schedule_b, C, 100)
    print(f"Learned weights: {linear_svm_b.weights[1:]}")
    print(f"Learned bias: {linear_svm_b.weights[0]}")
    print(f"Training accuracy: {np.mean(train_y == linear_svm_b.make_prediction(train_x))}")
    print(f"Testing accuracy: {np.mean(test_y == linear_svm_b.make_prediction(test_x))}")

print("Kernelized SVM, a)")
Cs = [100/873, 500/873, 700/873]
for C in Cs:
    print(f"C = {C}")
    kernelized_svm_a = vectormachine.KernelizedSVM(train_x, train_y, C)
    print(f"Learned weights: {kernelized_svm_a.weights}")
    print(f"Learned bias: {kernelized_svm_a.bias}")
    print(f"Training accuracy: {np.mean(train_y == kernelized_svm_a.make_prediction(train_x))}")
    print(f"Testing accuracy: {np.mean(test_y == kernelized_svm_a.make_prediction(test_x))}")


print("Kernelized SVM, b)")
gammas = [0.1, 0.5, 1, 5, 100]
support_vectors_overlap = []
for C in Cs:
    for gamma in gammas:
        print(f"C = {C}, gamma = {gamma}")
        kernelized_svm_b = vectormachine.KernelizedSVM(train_x, train_y, C, 'gaussian', gamma)
        print(f"Learned weights: {kernelized_svm_b.weights}")
        print(f"Learned bias: {kernelized_svm_b.bias}")
        print(f"Number of support vectors: {len(kernelized_svm_b.support_vectors)}")
        if C == 500/873:
            support_vectors_overlap.append(kernelized_svm_b.support_vectors)
        print(f"Training accuracy: {np.mean(train_y == kernelized_svm_b.make_prediction(train_x, 'gaussian', gamma))}")
        print(f"Testing accuracy: {np.mean(test_y == kernelized_svm_b.make_prediction(test_x, 'gaussian', gamma))}")

if C == 500/873:
    for i in range(len(gammas) - 1):
        overlap_count = 0
        for sv in support_vectors_overlap[i]:
            if sv in support_vectors_overlap[i + 1]:
                overlap_count += 1
        print(f"Overlap from gamma = {gammas[i]} to {gammas[i+1]}: {overlap_count}")


# In[ ]:





# In[ ]:





# In[ ]:




