#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import LMSRegression

train_data = pd.read_csv('concrete_train.csv', header=None) 
test_data = pd.read_csv('concrete_test.csv', header=None) 
X_train = train_data.iloc[:, 0:7].to_numpy()
y_train = train_data.iloc[:, 7].to_numpy()
X_test = test_data.iloc[:, 0:7].to_numpy()
y_test = test_data.iloc[:, 7].to_numpy()

#Batch gradient descent
batch_LMS_model = LMSRegression()
learning_rate = 0.01
norm_w_diff_thresh = 1e-6
cost_vals = batch_LMS_model.train_batch_gradient_descent(X_train, y_train, r=learning_rate, norm_w_diff_thresh=norm_w_diff_thresh)
batch_final_w = batch_LMS_model.w
batch_test_data_cost = LMSRegression.cost(X_test, y_test, batch_final_w)

#Plot cost
plt.figure()
plt.plot(cost_vals, label='Training')
plt.title('Batch Gradient Descent')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.show()

#Display final weight vector and final cost on training data
print('Batch gradient descent and a learning rate of r = {0} the final weight vector is {1}.'.format(learning_rate, batch_final_w))
print('The cost was {0:.3f}.'.format(batch_test_data_cost))


#Stochastic gradient descent
stochastic_LMS_model = LMSRegression()
learning_rate = 0.01
abs_cost_diff_thresh = 1e-6
cost_vals = stochastic_LMS_model.train_stochastic_gradient_descent(X_train, y_train, r=learning_rate, abs_cost_diff_thresh=abs_cost_diff_thresh)
stochastic_final_w = stochastic_LMS_model.w
stochastic_test_data_cost = LMSRegression.cost(X_test, y_test, stochastic_final_w)

#Plot cost 
plt.figure()
plt.plot(cost_vals, label='Cost')
plt.title('Stochastic Gradient Descent')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.show()

#Display final weight vector and final cost on training data
print('Stochastic gradient descent and a learning rate of r = {0} the final weight vector is {1}.'.format(learning_rate, stochastic_final_w))
print('The cost using the learned weight vector on the test data was {0:.3f}.'.format(stochastic_test_data_cost))


#Analytical weight vector solution
analytical_LMS_model = LMSRegression()
analytical_LMS_model.train_analytical(X_train, y_train)
analytical_final_w = analytical_LMS_model.w
analytical_test_data_cost = LMSRegression.cost(X_test, y_test, analytical_final_w)

#Display final weight vector and final cost on training data
print('Using the direct analytical weight vector solution for LMS the final weight vector is {0}.'.format(analytical_final_w))
print('The cost using the learned weight vector on the test data was {0:.3f}.'.format(analytical_test_data_cost))


print()
print('Batch w: {0}, cost: {1}\n Stochastic w: {2}, cost: {3}\n Analytical w: {4}, cost: {5}\n'.format(batch_final_w, batch_test_data_cost, stochastic_final_w, stochastic_test_data_cost, analytical_final_w, analytical_test_data_cost))


# In[ ]:




