#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import DT_Weighted as dt
import math
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

#load test data
test_data =  pd.read_csv('bank_test.csv', dtype=types)
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
        'balance': [0, 1],  # converted to binary
        'housing': ['yes', 'no'],
        'loan': ['yes', 'no'],
        'contact': ['unknown', 'telephone', 'cellular'],
        'day': [0, 1],  # converted to binary,
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'duration': [0, 1],  # converted to binary
        'campaign': [0, 1],  # converted to binary
        'days': [0, 1],  # converted to binary
        'previous': [0, 1],  # converted to binary
        'outcome': ['unknown', 'other', 'failure', 'success']}
label = {'Y':['yes','no']}


T = 500
train_size = len(train_data.index)
test_size = len(test_data.index)

alphas = [0 for x in range(T)]

weights = np.array([1 / train_size for x in range(train_size)])

train_err = [0 for x in range(T)]
test_err = [0 for x in range(T)]

train_errT = [0 for x in range(T)]
test_errT = [0 for x in range(T)]

train_r = [0 for x in range(T)]
test_r = [0 for x in range(T)]

train_py = np.array([0 for x in range(train_size)])
test_py = np.array([0 for x in range(train_size)])
for t in range(T):

    # ID3 stumps
    dt_generator = dt.WID3(feature_selection=0, max_depth=1)
    decision_tree = dt_generator.generate_decision_tree(train_data, features, label, weights)
    
    # training error
    train_data['py']= dt_generator.classify(decision_tree, train_data)
    tmp = train_data.apply(lambda row: 1 if row['Y']== row['py'] else 0, axis=1) 
    err = 1 - tmp.sum() / train_size
    train_err[t] = err
    
    # calculate weighted error and alpha
    tmp = train_data.apply(lambda row: 1 if row['Y']== row['py'] else -1, axis=1) 
    tmp = np.array(tmp.tolist())
    w = weights[tmp == -1]
    err = np.sum(w)
    #print('weighted error: ', err)
    alpha = 0.5 * math.log((1 - err) / err)
    alphas[t] = alpha

    # new weights
    weights = np.exp(tmp * -alpha) * weights
    total = np.sum(weights)
    weights = weights / total
    
    # testing error
    test_data['py']= dt_generator.classify(decision_tree, test_data)
    tmp = test_data.apply(lambda row: 1 if row['Y']== row['py'] else 0, axis=1) 
    test_err[t] = 1 - tmp.sum() / test_size

    # combined prediction
    # train
    py = np.array(train_data['py'].tolist())
    py[py == 'yes'] = 1
    py[py == 'no'] = -1
    py = py.astype(int)
    train_py = train_py + py * alpha
    py = py.astype(str)
    py[train_py > 0] = 'yes'
    py[train_py <=0] = 'no'
    train_data['py'] = pd.Series(py)
    tmp = train_data.apply(lambda row: 1 if row['Y']== row['py'] else 0, axis=1) 
    err = 1 - tmp.sum() / train_size
    train_errT[t] = err

    # test
    py = np.array(test_data['py'].tolist())
    py[py == 'yes'] = 1
    py[py == 'no'] = -1
    py = py.astype(int)
    test_py = test_py + py * alpha
    py = py.astype(str)
    py[test_py > 0] = 'yes'
    py[test_py <=0] = 'no'
    test_data['py'] = pd.Series(py)
    tmp = test_data.apply(lambda row: 1 if row['Y']== row['py'] else 0, axis=1) 
    err = 1 - tmp.sum() / test_size
    test_errT[t] = err
    print('t: ', t, 'train_t_err: ', train_err[t], 'test_t_err: ', test_err[t], 'train_T_err', train_errT[t], 'test_T_err:', test_errT[t])
import matplotlib.pyplot as plt
# Figure 1
f1, ax1 = plt.subplots()
ax1.plot(train_err, 'g', linewidth=2)  
ax1.plot(test_err, 'm', linewidth=2)   
ax1.legend(['Training Error', 'Test Error'], fontsize=12)
ax1.set_title('Decision Stumps Learned in Each Iteration', fontsize=16)
ax1.set_xlabel('Iteration', fontsize=14)
ax1.set_ylabel('Error', fontsize=14)


# Figure 2
f2, ax2 = plt.subplots()
ax2.plot(train_errT, 'c', linewidth=2)  
ax2.plot(test_errT, 'y', linewidth=2)  
ax2.legend(['Training Error', 'Test Error'], fontsize=12)
ax2.set_title('Training and Test errors variation along with T', fontsize=16)
ax2.set_xlabel('Iteration', fontsize=14)
ax2.set_ylabel('Error', fontsize=14)


# Show the plots
plt.show()

