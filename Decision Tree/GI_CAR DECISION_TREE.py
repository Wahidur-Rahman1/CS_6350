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


def gini_index(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    gini = 1 - np.sum([(counts[i]/np.sum(counts)) ** 2 for i in range(len(elements))])
    return gini

def gini_gain(data, split_attribute, target_name="label"):
    total_gini = gini_index(data[target_name])
    
    vals, counts = np.unique(data[split_attribute], return_counts=True)
    weighted_gini = sum([(counts[i]/np.sum(counts)) * gini_index(data.where(data[split_attribute]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    gini_reduction = total_gini - weighted_gini
    return gini_reduction


def ID3(data, original_data, features, target_attribute_name="label", max_depth=None, tree=None, depth=0):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
    elif len(features) == 0 or (max_depth and depth >= max_depth):
        return np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
    else:
        depth += 1
        item_values = [gini_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        tree = {best_feature: {}}
        
        features = [i for i in features if i != best_feature]
        
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data, original_data, features, target_attribute_name, max_depth, tree, depth)
            tree[best_feature][value] = subtree
        
        return tree
    
train_data = pd.read_csv('car_train.csv')
tree = ID3(train_data, train_data, train_data.columns[:-1], max_depth=4)
print(tree)
   


# In[ ]:


def predict(tree, sample):
    for attribute in tree.keys():
        value = sample[attribute]
        if value in tree[attribute]:
            tree = tree[attribute][value]
            prediction = 0
            if type(tree) is dict:
                prediction = predict(tree, sample)
            else:
                prediction = tree
                break;
        else:
            return None
    return prediction

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


train_error = evaluate(tree, train_data)
print(f"Training error: {train_error:.3f}")

test_data = pd.read_csv('car_test.csv')
test_error = evaluate(tree, test_data)
print(f"Test error: {test_error:.3f}")


# In[ ]:




