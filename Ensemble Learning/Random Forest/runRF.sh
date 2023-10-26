#!/bin/bash

# Upgrade scikit-learn, scipy, matplotlib, and pandas
pip install -U scikit-learn scipy matplotlib pandas


# Run first file
echo "Decision_Tree_RF.py" 
python3  Decision_Tree_RF.py


# Run 2nd file
echo "Running  Bank_RF_2.py"
python3 Bank_RF_2.py


# Run 3rd file
echo "Running  Bank_RF_4.py"
python3 Bank_RF_4.py


# Run 4th file
echo "Running  Bank_RF_6.py"
python3 Bank_RF_6.py

# Run 5th file
echo "Running  credit_rf_2.py"
python3 credit_rf_2.py

# Run 6th file
echo "Running  credit_rf_4.py"
python3 credit_rf_4.py


# Run 7th file
echo "Running  credit_rf_6.py"
python3 credit_rf_6.py
