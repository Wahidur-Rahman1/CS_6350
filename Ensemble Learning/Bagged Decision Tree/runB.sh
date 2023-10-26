#!/bin/bash

# Upgrade scikit-learn, scipy, matplotlib, and pandas
pip install -U scikit-learn scipy matplotlib pandas


# Run first file
echo "Decisionn_Tree.py" 
python3  Decisionn_Tree.py


# Run 2nd file
echo "Running  Bank_BaggedDT_final.py"
python3 Bank_BaggedDT_final.py


# Run 3rd file
echo "Running  Bagged_Credit_DT.py"
python3 Bagged_Credit_DT.py
