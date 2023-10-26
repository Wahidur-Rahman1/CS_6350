#!/bin/bash

# Upgrade scikit-learn, scipy, matplotlib, and pandas
pip install -U scikit-learn scipy matplotlib pandas


# Run first file
echo "Decisionn_Tree.py" 
python3  Decisionn_Tree.py


# Run 2nd file
echo "Running  100_bagged_tree.py"
python3 100_bagged_tree.py
