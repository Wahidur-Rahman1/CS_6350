#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

train_original = pd.read_csv('bank_train.csv')
df_test = pd.read_csv('bank_test.csv')


# In[ ]:


# Determine the most common value (mode) for the 'outcome' column excluding 'unknown'
majority_value = train_original[train_original['outcome'] != 'unknown']['outcome'].mode().iloc[0]

# Replace 'unknown' values with the majority value
train_original['outcome'].replace('unknown', majority_value, inplace=True)

# Determine the most common value (mode) for the 'outcome' column excluding 'unknown'
majority_value = df_test[df_test['outcome'] != 'unknown']['outcome'].mode().iloc[0]

# Replace 'unknown' values with the majority value
df_test['outcome'].replace('unknown', majority_value, inplace=True)


# In[ ]:


train_original.head()


# In[ ]:


def is_numeric(value):
    return isinstance(value, (int, float, np.int64, np.float64))

def binarize_data(data, feature):
    if is_numeric(data[feature].iloc[0]):
        median_val = data[feature].median()
        data[feature] = np.where(data[feature] > median_val, "> " + str(median_val), "<= " + str(median_val))
    return data


columns_to_binarize = ['age', 'balance', 'day', 'duration', 'campaign', 'days', 'previous']

for col in columns_to_binarize:
    train_original = binarize_data(train_original, col)
    
for col in columns_to_binarize:
    df_test = binarize_data(df_test, col)    
    

print("\nDataFrame after binarizing columns':")
print(train_original)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

columns_to_encode = ['job', 'marital', 'education', 'contact', 'month', 'outcome','Y','default','housing','loan','age','balance','day','duration','campaign', 'days', 'previous']

for column in columns_to_encode:
    train_original[column] = le.fit_transform(train_original[column])
    
for column in columns_to_encode:
    df_test[column] = le.fit_transform(df_test[column])   


# In[ ]:


from sklearn.model_selection import train_test_split
from copy import deepcopy
import numpy as np

# Helper Functions

def gini_index(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    gini = 1 - np.sum([(counts[i]/np.sum(counts)) ** 2 for i in range(len(elements))])
    return gini

def gini_gain(data, split_attribute, target_name="Y"):
    total_gini = gini_index(data[target_name])
    
    vals, counts = np.unique(data[split_attribute], return_counts=True)
    weighted_gini = sum([(counts[i]/np.sum(counts)) * gini_index(data.where(data[split_attribute]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    gini_reduction = total_gini - weighted_gini
    return gini_reduction

# Pruning

def predict(tree, sample):
    """Recursively traverse the decision tree to predict the class label for a single sample."""
    for attribute, sub_tree in tree.items():
        if sample[attribute] in sub_tree.keys():
            if isinstance(sub_tree[sample[attribute]], dict):  # If nested dictionary, traverse
                return predict(sub_tree[sample[attribute]], sample)
            else:
                return sub_tree[sample[attribute]]
    return None

def compute_accuracy(tree, data, target_attribute_name):
    """Compute accuracy of the decision tree on given data."""
    predictions = data.apply(lambda x: predict(tree, x), axis=1)
    correct_predictions = (predictions == data[target_attribute_name]).sum()
    accuracy = correct_predictions / len(data)
    return accuracy


def reduced_error_pruning(tree, validation_data, target_attribute_name):
    if not isinstance(tree, dict):
        return tree

    for attribute in tree:
        for value in tree[attribute]:
            pruned_tree = most_common_class(validation_data, target_attribute_name)
            original_tree = deepcopy(tree)
            
            tree[attribute][value] = pruned_tree
            original_accuracy = compute_accuracy(original_tree, validation_data, target_attribute_name)
            pruned_accuracy = compute_accuracy(tree, validation_data, target_attribute_name)

            if original_accuracy > pruned_accuracy:
                tree[attribute][value] = original_tree[attribute][value]
            else:
                tree[attribute][value] = reduced_error_pruning(tree[attribute][value], validation_data, target_attribute_name)
                
    return tree

def most_common_class(data, target_attribute_name):
    return data[target_attribute_name].value_counts().idxmax()

# ID3 Algorithm

def ID3(data, original_data, features, target_attribute_name="Y", max_depth=None, tree=None, depth=0):
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

train_data, validation_data = train_test_split(train_original, test_size=0.2, random_state=42)

# Build and prune the tree
tree = ID3(train_data, train_data, train_data.columns[:-1], max_depth=8) 
pruned_tree = reduced_error_pruning(tree, validation_data, 'Y')

print(pruned_tree)


# In[ ]:


def evaluate(pruned_tree, data):
    correct_predictions = 0
    for i in range(len(data)):
        prediction = predict(pruned_tree, data.iloc[i])
        if prediction == data.iloc[i]['Y']:
            correct_predictions += 1

    accuracy = correct_predictions / len(data)
    error = 1 - accuracy
    return error


# In[ ]:


# Assuming you have loaded train_data and test_data
train_error = evaluate(pruned_tree, train_original)
print(f"Training error: {train_error:.3f}")


test_error = evaluate(pruned_tree, df_test)
print(f"Test error: {test_error:.3f}")


# In[ ]:




