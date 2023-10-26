#!/bin/bash

# Upgrade scikit-learn, scipy, matplotlib, and pandas
pip install -U scikit-learn scipy matplotlib pandas


# Run first file
echo "DT_Weighted.py" 
python3  DT_Weighted.py


# Run 2nd file
echo "Running  AdaBoost_Bank_Final.py"
python3 AdaBoost_Bank_Final.py


# Run 3rd file
echo "Running  Credit_adaboost.py"
python3 Credit_adaboost.py
