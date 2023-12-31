
import pandas as pd
import math
import copy
import numpy as np

class TreeNode:
    def __init__(self):
        self.feature = None
        self.children = None
        self.depth = -1
        self.isLeaf = False
        self.label = None
    
    def set_feature(self, feature):
        self.feature = feature

    def set_children(self, children):
        self.children = children
    
    def get_depth(self):
        return self.depth

    def set_depth(self, depth):
        self.depth = depth

    def set_leaf(self):
        self.isLeaf = True

    def set_label(self, label):
        self.label = label

    def is_leaf(self):
        return self.isLeaf



class WID3:

    def __init__(self, feature_selection = 0, max_depth = 10):
        self.feature_selection = feature_selection
        self.max_depth = max_depth
        

    def set_feature_selection(self, feature_selection):
        self.feature_selection = feature_selection
    
    def set_max_depth(self, max_depth):
        self.max_depth = max_depth
    

    def get_E(self, dataset, label, weights):
        ln, lv = list(label.items())[0]
        total = np.sum(weights)
        column = np.array(dataset[ln].tolist())
        # total = len(dataset.index)
        if total == 0:
            return 0
        entropy = 0

        for v in lv:
            w = weights[column == v]
            p = np.sum(w) / total
            # p = len(dataset[dataset[ln] == v]) / total
            if p != 0:
                entropy += -p * math.log2(p)
        return entropy
    
    def get_ME(self, dataset, label, weights):
        ln, lv = list(label.items())[0]
        total = np.sum(weights)
        column = np.array(dataset[ln].tolist())
        # total = len(dataset.index)
        if total == 0:
            return 0
        max_p = 0
        for v in lv:
            w = weights[column == v]
            p = np.sum(w) / total
            # p = len(dataset[dataset[ln] == v]) / total
            max_p = max(max_p, p)
        return 1 - max_p
        
    
    def get_GI(self, dataset, label, weights):
        ln, lv = list(label.items())[0]
        # total = len(dataset.index)
        total = np.sum(weights)
        column = np.array(dataset[ln].tolist())
        if total == 0:
            return 0
        sqrsum = 0
        for v in lv:
            w = weights[column == v]
            p = np.sum(w) / total
            # p = len(dataset[dataset[ln] == v]) / total
            sqrsum += p**2
        return 1 - sqrsum
    
    def get_majority(self, dataset, label, weights):
        ln, lv = list(label.items())[0]
        majority_label = None
        max_sum = -1
        column = np.array(dataset[ln].tolist())
        for v in lv:
            w = weights[column == v]
            weight_sum = np.sum(w)
            if weight_sum > max_sum:
                majority_label = v
                max_sum = weight_sum
        
        return majority_label

    def split_dataset(self, cur):
        nodeList = []
        
        features = cur['features']
        label = cur['label']
        dtNode = cur['dtNode']
        dataset = cur['dataset']
        weights = cur['weights']

        measure = None
        if self.feature_selection == 0:
            measure = self.get_E
        elif self.feature_selection == 1:
            measure = self.get_ME
        elif self.feature_selection == 2:
            measure = self.get_GI
            
        total = sum(weights)
        majority_label = self.get_majority(dataset, label, weights)


        stat = measure(dataset, label, weights)
        # pure or achieve max depth or no remaining features
        if stat == 0 or dtNode.get_depth() == self.max_depth or len(features.items()) == 0:
            dtNode.set_leaf()
            if total > 0:
                dtNode.set_label(majority_label)
            return nodeList

        max_gain = -1
        max_fn = None
        # select feature which results in maximum gain
        for fn, fv in features.items():
            column = np.array(dataset[fn].tolist())
            gain = 0
            for v in fv:
                w = weights[column == v]
                sub_weights = w
                p = np.sum(sub_weights) / total
                subset = dataset[dataset[fn] == v]
                # p = len(subset.index) / total
                gain += p * measure(subset, label, sub_weights)
            gain = stat - gain
            if gain > max_gain:
                max_gain = gain
                max_fn = fn
        
        children = {}
        dtNode.set_feature(max_fn)
        # remaining features
        rf = copy.deepcopy(features)
        rf.pop(max_fn, None)
        # split dataset
        column = np.array(dataset[max_fn].tolist())
        for v in features[max_fn]:
            childNode = TreeNode()
            childNode.set_depth(dtNode.get_depth() + 1)
            childNode.set_label(majority_label)
            children[v] = childNode
            w = weights[column == v]
            pNode = {'dataset': dataset[dataset[max_fn] == v],'weights': w ,'features':  copy.deepcopy(rf), 'label': label, 'dtNode': childNode}
            nodeList.append(pNode)
        # set children of decision node
        dtNode.set_children(children)
        # return processing nodes
        return nodeList
    def generate_decision_tree(self, dataset, features, label, weights):
        Q = []
        dtRoot = TreeNode()
        dtRoot.set_depth(0)
        # processing node root
        root = {'dataset': dataset,'weights': weights, 'features': features, 'label': label, 'dtNode': dtRoot}
        Q.append(root)
        while len(Q) > 0:
            cur = Q.pop(0)
            nodes = self.split_dataset(cur)
            for node in nodes:
                Q.append(node)
        return dtRoot
    

    def classifyOneInstance(self, dt, test_data):
        p = dt
        while not p.is_leaf(): 
            p = p.children[test_data[p.feature]]
        return p.label

    def classify(self, dt, test_data):
        return test_data.apply(lambda row: self.classifyOneInstance(dt, row), axis=1)





