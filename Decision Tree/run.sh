#!/bin/bash

# Upgrade scikit-learn, scipy, matplotlib, and pandas
pip install -U scikit-learn scipy matplotlib pandas


# Run first file
echo "IG_CAR_DECISION_TREE.py" 
python3  IG_CAR_DECISION_TREE.py



# Run 2nd file
echo "Running  GI_CAR_DECISION_TREE.py"
python3 GI_CAR_DECISION_TREE.py


# Run 3rd file
echo "Running  ME_CAR_DECISION_TREE.py"
python3 ME_CAR_DECISION_TREE.py

# Run 4th file
echo "IG_pruned_bank.py" 
python3  IG_pruned_bank.py



# Run 5th file
echo "Running  GI_pruned_bank.py"
python3  GI_pruned_bank.py


# Run 6th file
echo "Running   ME_pruned_bank.py"
python3  ME_pruned_bank.py

# Run 7th file
echo "IG_Pruned_missing_value.py" 
python3  IG_Pruned_missing_value.py


# Run 8th file
echo "Running  GI_Pruned_missing_value.py"
python3 GI_Pruned_missing_value.py


# Run 9th file
echo "Running  ME_Pruned_missing_value.py"
python3 ME_Pruned_missing_value.py




