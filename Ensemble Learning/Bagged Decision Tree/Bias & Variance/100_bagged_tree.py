#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import Decisionn_Tree as dt
import numpy as np
import matplotlib.pyplot as plt

columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'days', 'previous', 'outcome', 'Y']
types = {'age': int, 'job': str,'marital': str,'education': str,'default': str,'balance': int,'housing': str,'loan': str,'contact': str,'day': int,'month': str,'duration': int,'campaign': int,'days': int,'previous': int,'outcome': str,'Y': str}

# load train data
train_data =  pd.read_csv('bank_train.csv', dtype=types)
train_size = len(train_data.index)

# convert numeric to binary
numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'days', 'previous']
for c in numeric_features:
    median = train_data[c].median()
    train_data[c] = train_data[c].apply(lambda x: 0 if x < median else 1)

test_data =  pd.read_csv('bank_train.csv', dtype=types)
test_size = len(test_data.index)
for c in numeric_features:
    median = test_data[c].median()
    test_data[c] = test_data[c].apply(lambda x: 0 if x < median else 1)


# set features and label
features = {'age': [0, 1],  # converted to binary
        'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services'], 
        'marital': ['married','divorced','single'], 
        'education': ['unknown', 'secondary', 'primary', 'tertiary'],
        'default': ['yes', 'no'],
        'balance': [0, 1],  
        'housing': ['yes', 'no'],
        'loan': ['yes', 'no'],
        'contact': ['unknown', 'telephone', 'cellular'],
        'day': [0, 1], 
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'duration': [0, 1], 
        'campaign': [0, 1],  
        'days': [0, 1],  
        'previous': [0, 1], 
        'outcome': ['unknown', 'other', 'failure', 'success']}
label = {'Y': ['yes', 'no']}

T = 500
num_run = 100
test_py = np.array([[0 for x in range(test_size)] for y in range(num_run)])
test_py_first = np.array([0 for x in range(test_size)])

for iter in range(num_run):
        train_subset = train_data.sample(n=1000, replace=False, random_state=iter)
        print('iter: ', iter)
        for t in range(T):
            
                # sample with replace
                sampled = train_subset.sample(frac=0.01, replace=True, random_state=t)
                # ID3
                dt_generator = dt.ID3(feature_selection=0, max_depth=17)
                # get decision tree
                decision_tree = dt_generator.generate_decision_tree(sampled, features, label)
                ## predict
                # test
                py = dt_generator.classify(decision_tree, test_data) 
                py = np.array(py.tolist())
                py[py == 'yes'] = 1
                py[py == 'no'] = -1
                py = py.astype(int)
                test_py[iter] = test_py[iter] + py
                if t == 0:
                        test_py_first = test_py_first + py

true_value = np.array(test_data['Y'].tolist())
true_value[true_value == 'yes'] = 1
true_value[true_value == 'no'] = -1
true_value = true_value.astype(int)

# frist tree predictor
# take average
test_py_first = test_py_first / num_run
# bias
bias = np.mean(np.square(test_py_first - true_value))
# variance
mean = np.mean(test_py_first) 
variance = np.sum(np.square(test_py_first - mean)) / (test_size - 1)
se = bias + variance

print(bias)
print(variance)
print('100 single tree predictor: ', se)
# bagged tree predictor
# take average
test_py = np.sum(test_py,axis=0) / (num_run * T)
# bias
bias = np.mean(np.square(test_py - true_value))
# variance
mean = np.mean(test_py)
variance = np.sum(np.square(test_py - mean)) / (test_size - 1)
se = bias + variance

print(bias)
print(variance)
print('100 bagged tree predictor:', se)

