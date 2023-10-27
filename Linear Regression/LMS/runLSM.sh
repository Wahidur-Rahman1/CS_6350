#!/bin/bash

# Upgrade scikit-learn, scipy, matplotlib, and pandas
pip install -U scikit-learn scipy matplotlib pandas


# Run first file
echo "linear_regression.py" 
python3  linear_regression.py



# Run 2nd file
echo "Running  LMS_GD_final.py"
python3 LMS_GD_final.py
