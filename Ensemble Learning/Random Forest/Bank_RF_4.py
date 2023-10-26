#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import Decision_Tree_RF as dt
import numpy as np
import matplotlib.pyplot as plt

columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'days', 'previous', 'outcome', 'Y']
types = {'age': int, 'job': str,'marital': str,'education': str,'default': str,'balance': int,'housing': str,'loan': str,'contact': str,'day': int,'month': str,'duration': int,'campaign': int,'days': int,'previous': int,'outcome': str,'Y': str}

# load train data
train_data =  pd.read_csv('bank_train.csv', dtype=types)
train_size = len(train_data.index)
## process data
# convert numeric to binary
numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'days', 'previous']
for c in numeric_features:
    median = train_data[c].median()
    train_data[c] = train_data[c].apply(lambda x: 0 if x < median else 1)
    
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
        'duration': [0, 1],  
        'campaign': [0, 1], 
        'days': [0, 1],  
        'previous': [0, 1],  
        'outcome': ['unknown', 'other', 'failure', 'success']}
label = {'Y': ['yes', 'no']}

T = 500

train_err = [0 for x in range(T)]
test_err = [0 for x in range(T)]
train_py = np.array([0 for x in range(train_size)])
test_py = np.array([0 for x in range(test_size)])

for t in range(T):
    # sample with replace
    sampled = train_data.sample(frac=0.5, replace=True, random_state=t)
    # ID3
    dt_generator = dt.ID3(feature_selection=0, max_depth=17, subset=4)
    # get decision tree
    decision_tree = dt_generator.generate_decision_tree(sampled, features, label)

    ## predict
    # train
    py = dt_generator.classify(decision_tree, train_data) 
    py = np.array(py.tolist())
    py[py == 'yes'] = 1
    py[py == 'no'] = -1
    py = py.astype(int)
    train_py = train_py + py
    py = py.astype(str)
    py[train_py > 0] = 'yes'
    py[train_py <=0] = 'no'
    train_data['py'] = pd.Series(py)

    acc = train_data.apply(lambda row: 1 if row['Y'] == row['py'] else 0, axis=1).sum() / train_size
    err = 1 - acc
    train_err[t] = err
    # test
    py = dt_generator.classify(decision_tree, test_data) 
    py = np.array(py.tolist())
    py[py == 'yes'] = 1
    py[py == 'no'] = -1
    py = py.astype(int)
    test_py = test_py + py
    py = py.astype(str)
    py[test_py > 0] = 'yes'
    py[test_py <=0] = 'no'
    test_data['py'] = pd.Series(py)
    acc = test_data.apply(lambda row: 1 if row['Y'] == row['py'] else 0, axis=1).sum() / test_size
    err = 1 - acc
    test_err[t] = err
    print('t: ', t, 'train_err: ', train_err[t], 'test_err: ', test_err[t])


fig = plt.figure()
fig.suptitle('Feature Subset = 4')
plt.xlabel('Iteration', fontsize=18)
plt.ylabel('Error Rate', fontsize=16)
plt.plot(train_err, 'b')
plt.plot(test_err, 'r')  
plt.legend(['Training Error', 'Test Error'])

plt.show()


# In[ ]:




