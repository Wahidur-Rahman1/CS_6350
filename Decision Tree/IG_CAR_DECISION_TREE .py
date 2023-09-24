#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


train_data = pd.read_csv('car_train.csv')


# In[ ]:


test_data = pd.read_csv('car_test.csv')


# In[ ]:


test_data.head()


# In[ ]:


train_data.head()


# In[ ]:


import numpy as np
import pandas as pd

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = -np.sum([(counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def info_gain(data, split_attribute, target_name="label"):
    total_entropy = entropy(data[target_name])
    
    vals, counts = np.unique(data[split_attribute], return_counts=True)
    weighted_entropy = sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    information_gain = total_entropy - weighted_entropy
    return information_gain

def ID3(data, original_data, features, target_attribute_name="label", max_depth=None, tree=None, depth=0):
    # Base cases
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
    
    elif len(features) == 0:
        return np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
    
    elif max_depth and depth >= max_depth:
        return np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
    
    else:
        depth += 1
        item_values = [info_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        tree = {best_feature: {}}
        
        features = [i for i in features if i != best_feature]
        
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data, original_data, features, target_attribute_name, max_depth, tree, depth)
            tree[best_feature][value] = subtree
        
        return tree

# To use
train_data = pd.read_csv('car_train.csv')
tree = ID3(train_data, train_data, train_data.columns[:-1], max_depth=6) # Assuming last column is target, set max depth to 3
print(tree)


# In[ ]:


def predict(tree, sample):
    for attribute in tree.keys():
        value = sample[attribute]  # Removed .item()
        if value in tree[attribute]:
            tree = tree[attribute][value]
            prediction = 0
        
            if type(tree) is dict:
                prediction = predict(tree, sample)
            else:
                prediction = tree
                break
        else:  # Case when the tree doesn't have a branch for the given value.
            return None  # or any other default action
    
    return prediction


# In[ ]:


def evaluate(tree, data):
    correct_predictions = 0
    for i in range(len(data)):
        prediction = predict(tree, data.iloc[i])
        if prediction == data.iloc[i]['label']:
            correct_predictions += 1

    accuracy = correct_predictions / len(data)
    error = 1 - accuracy
    return error


# In[ ]:


# Assuming you have loaded train_data and test_data
train_error = evaluate(tree, train_data)
print(f"Training error: {train_error:.3f}")


test_error = evaluate(tree, test_data)
print(f"Test error: {test_error:.3f}")


# In[ ]:




