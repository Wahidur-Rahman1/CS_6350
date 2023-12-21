#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import glob

import os
path1 = './Data_MRR'
file_list1 = glob.glob(path1 + "/*.csv")
data1 = []
for file in file_list1:
    df1 = pd.read_csv(file)
    data1.append(df1)

combined_data1 = pd.concat(data1, ignore_index=True)


# In[44]:


input_parameters1= combined_data1[['MACHINE_ID','MACHINE_DATA','TIMESTAMP','WAFER_ID','STAGE','CHAMBER','USAGE_OF_BACKING_FILM', 'USAGE_OF_DRESSER','USAGE_OF_POLISHING_TABLE','USAGE_OF_DRESSER_TABLE','PRESSURIZED_CHAMBER_PRESSURE','MAIN_OUTER_AIR_BAG_PRESSURE','CENTER_AIR_BAG_PRESSURE','RETAINER_RING_PRESSURE','RIPPLE_AIR_BAG_PRESSURE','USAGE_OF_MEMBRANE','USAGE_OF_PRESSURIZED_SHEET','SLURRY_FLOW_LINE_A','SLURRY_FLOW_LINE_B','SLURRY_FLOW_LINE_C','WAFER_ROTATION','STAGE_ROTATION','HEAD_ROTATION','DRESSING_WATER_STATUS','EDGE_AIR_BAG_PRESSURE']].values


# In[45]:


y_train1 = pd.read_csv('./MRR.csv')
output_parameters1 =y_train1[['WAFER_ID','STAGE','AVG_REMOVAL_RATE']].values
import pandas as pd
import numpy as np

input_parameters1= combined_data1[['MACHINE_ID','MACHINE_DATA','TIMESTAMP','WAFER_ID','STAGE','CHAMBER','USAGE_OF_BACKING_FILM', 'USAGE_OF_DRESSER','USAGE_OF_POLISHING_TABLE','USAGE_OF_DRESSER_TABLE','PRESSURIZED_CHAMBER_PRESSURE','MAIN_OUTER_AIR_BAG_PRESSURE','CENTER_AIR_BAG_PRESSURE','RETAINER_RING_PRESSURE','RIPPLE_AIR_BAG_PRESSURE','USAGE_OF_MEMBRANE','USAGE_OF_PRESSURIZED_SHEET','SLURRY_FLOW_LINE_A','SLURRY_FLOW_LINE_B','SLURRY_FLOW_LINE_C','WAFER_ROTATION','STAGE_ROTATION','HEAD_ROTATION','DRESSING_WATER_STATUS','EDGE_AIR_BAG_PRESSURE']].values
output_parameters1 =y_train1[['WAFER_ID','STAGE','AVG_REMOVAL_RATE']].values

# Repeat output values accordingly to match input rows
repeated_output_values1 = np.repeat(output_parameters1, 1, axis=0)

# Create DataFrames for input and output
input_df1 = pd.DataFrame(input_parameters1, columns=['MACHINE_ID','MACHINE_DATA','TIMESTAMP','WAFER_ID','STAGE','CHAMBER','USAGE_OF_BACKING_FILM', 'USAGE_OF_DRESSER','USAGE_OF_POLISHING_TABLE','USAGE_OF_DRESSER_TABLE','PRESSURIZED_CHAMBER_PRESSURE','MAIN_OUTER_AIR_BAG_PRESSURE','CENTER_AIR_BAG_PRESSURE','RETAINER_RING_PRESSURE','RIPPLE_AIR_BAG_PRESSURE','USAGE_OF_MEMBRANE','USAGE_OF_PRESSURIZED_SHEET','SLURRY_FLOW_LINE_A','SLURRY_FLOW_LINE_B','SLURRY_FLOW_LINE_C','WAFER_ROTATION','STAGE_ROTATION','HEAD_ROTATION','DRESSING_WATER_STATUS','EDGE_AIR_BAG_PRESSURE'])
output_df1 = pd.DataFrame(repeated_output_values1, columns=['WAFER_ID','STAGE','AVG_REMOVAL_RATE'])

# Merge input and output DataFrames based on ID column
df1 = pd.merge(input_df1, output_df1, on=['WAFER_ID','STAGE'])


# Optionally, assign an index to each row
df1.index = range(1, len(df1) + 1)


# In[46]:


df1.head()


# In[47]:


df1.shape


# In[48]:


path2 = './validation'
file_list2 = glob.glob(path2 + "/*.csv")
data2 = []
for file in file_list2:
    df2 = pd.read_csv(file)
    data2.append(df2)

combined_data2 = pd.concat(data2, ignore_index=True)


# In[49]:


input_parameters2= combined_data2[['MACHINE_ID','MACHINE_DATA','TIMESTAMP','WAFER_ID','STAGE','CHAMBER','USAGE_OF_BACKING_FILM', 'USAGE_OF_DRESSER','USAGE_OF_POLISHING_TABLE','USAGE_OF_DRESSER_TABLE','PRESSURIZED_CHAMBER_PRESSURE','MAIN_OUTER_AIR_BAG_PRESSURE','CENTER_AIR_BAG_PRESSURE','RETAINER_RING_PRESSURE','RIPPLE_AIR_BAG_PRESSURE','USAGE_OF_MEMBRANE','USAGE_OF_PRESSURIZED_SHEET','SLURRY_FLOW_LINE_A','SLURRY_FLOW_LINE_B','SLURRY_FLOW_LINE_C','WAFER_ROTATION','STAGE_ROTATION','HEAD_ROTATION','DRESSING_WATER_STATUS','EDGE_AIR_BAG_PRESSURE']].values


# In[50]:


y_train2 = pd.read_csv('./orig_CMP_validation_removalrate.csv')
output_parameters2 =y_train2[['WAFER_ID','STAGE','AVG_REMOVAL_RATE']].values
import pandas as pd
import numpy as np

input_parameters2= combined_data2[['MACHINE_ID','MACHINE_DATA','TIMESTAMP','WAFER_ID','STAGE','CHAMBER','USAGE_OF_BACKING_FILM', 'USAGE_OF_DRESSER','USAGE_OF_POLISHING_TABLE','USAGE_OF_DRESSER_TABLE','PRESSURIZED_CHAMBER_PRESSURE','MAIN_OUTER_AIR_BAG_PRESSURE','CENTER_AIR_BAG_PRESSURE','RETAINER_RING_PRESSURE','RIPPLE_AIR_BAG_PRESSURE','USAGE_OF_MEMBRANE','USAGE_OF_PRESSURIZED_SHEET','SLURRY_FLOW_LINE_A','SLURRY_FLOW_LINE_B','SLURRY_FLOW_LINE_C','WAFER_ROTATION','STAGE_ROTATION','HEAD_ROTATION','DRESSING_WATER_STATUS','EDGE_AIR_BAG_PRESSURE']].values
output_parameters =y_train2[['WAFER_ID','STAGE','AVG_REMOVAL_RATE']].values

# Repeat output values accordingly to match input rows
repeated_output_values2 = np.repeat(output_parameters2, 1, axis=0)

# Create DataFrames for input and output
input_df2 = pd.DataFrame(input_parameters2, columns=['MACHINE_ID','MACHINE_DATA','TIMESTAMP','WAFER_ID','STAGE','CHAMBER','USAGE_OF_BACKING_FILM', 'USAGE_OF_DRESSER','USAGE_OF_POLISHING_TABLE','USAGE_OF_DRESSER_TABLE','PRESSURIZED_CHAMBER_PRESSURE','MAIN_OUTER_AIR_BAG_PRESSURE','CENTER_AIR_BAG_PRESSURE','RETAINER_RING_PRESSURE','RIPPLE_AIR_BAG_PRESSURE','USAGE_OF_MEMBRANE','USAGE_OF_PRESSURIZED_SHEET','SLURRY_FLOW_LINE_A','SLURRY_FLOW_LINE_B','SLURRY_FLOW_LINE_C','WAFER_ROTATION','STAGE_ROTATION','HEAD_ROTATION','DRESSING_WATER_STATUS','EDGE_AIR_BAG_PRESSURE'])
output_df2 = pd.DataFrame(repeated_output_values2, columns=['WAFER_ID','STAGE','AVG_REMOVAL_RATE'])

# Merge input and output DataFrames based on ID column
df2 = pd.merge(input_df2, output_df2, on=['WAFER_ID','STAGE'])


# Optionally, assign an index to each row
df2.index = range(1, len(df2) + 1)


# In[51]:


df2.head()


# In[52]:


df2.shape


# In[53]:


path3 = './test_dataset_cmp'
file_list3 = glob.glob(path3 + "/*.csv")
data3 = []
for file in file_list3:
    df3 = pd.read_csv(file)
    data3.append(df3)

combined_data3 = pd.concat(data3, ignore_index=True)


# In[54]:


input_parameters3= combined_data3[['MACHINE_ID','MACHINE_DATA','TIMESTAMP','WAFER_ID','STAGE','CHAMBER','USAGE_OF_BACKING_FILM', 'USAGE_OF_DRESSER','USAGE_OF_POLISHING_TABLE','USAGE_OF_DRESSER_TABLE','PRESSURIZED_CHAMBER_PRESSURE','MAIN_OUTER_AIR_BAG_PRESSURE','CENTER_AIR_BAG_PRESSURE','RETAINER_RING_PRESSURE','RIPPLE_AIR_BAG_PRESSURE','USAGE_OF_MEMBRANE','USAGE_OF_PRESSURIZED_SHEET','SLURRY_FLOW_LINE_A','SLURRY_FLOW_LINE_B','SLURRY_FLOW_LINE_C','WAFER_ROTATION','STAGE_ROTATION','HEAD_ROTATION','DRESSING_WATER_STATUS','EDGE_AIR_BAG_PRESSURE']].values


# In[55]:


y_train3 = pd.read_csv('./orig_CMP_test_removalrate.csv')
output_parameters3 =y_train3[['WAFER_ID','STAGE','AVG_REMOVAL_RATE']].values
import pandas as pd
import numpy as np

input_parameters3= combined_data3[['MACHINE_ID','MACHINE_DATA','TIMESTAMP','WAFER_ID','STAGE','CHAMBER','USAGE_OF_BACKING_FILM', 'USAGE_OF_DRESSER','USAGE_OF_POLISHING_TABLE','USAGE_OF_DRESSER_TABLE','PRESSURIZED_CHAMBER_PRESSURE','MAIN_OUTER_AIR_BAG_PRESSURE','CENTER_AIR_BAG_PRESSURE','RETAINER_RING_PRESSURE','RIPPLE_AIR_BAG_PRESSURE','USAGE_OF_MEMBRANE','USAGE_OF_PRESSURIZED_SHEET','SLURRY_FLOW_LINE_A','SLURRY_FLOW_LINE_B','SLURRY_FLOW_LINE_C','WAFER_ROTATION','STAGE_ROTATION','HEAD_ROTATION','DRESSING_WATER_STATUS','EDGE_AIR_BAG_PRESSURE']].values
output_parameters3 =y_train3[['WAFER_ID','STAGE','AVG_REMOVAL_RATE']].values

# Repeat output values accordingly to match input rows
repeated_output_values3 = np.repeat(output_parameters3, 1, axis=0)

# Create DataFrames for input and output
input_df3 = pd.DataFrame(input_parameters3, columns=['MACHINE_ID','MACHINE_DATA','TIMESTAMP','WAFER_ID','STAGE','CHAMBER','USAGE_OF_BACKING_FILM', 'USAGE_OF_DRESSER','USAGE_OF_POLISHING_TABLE','USAGE_OF_DRESSER_TABLE','PRESSURIZED_CHAMBER_PRESSURE','MAIN_OUTER_AIR_BAG_PRESSURE','CENTER_AIR_BAG_PRESSURE','RETAINER_RING_PRESSURE','RIPPLE_AIR_BAG_PRESSURE','USAGE_OF_MEMBRANE','USAGE_OF_PRESSURIZED_SHEET','SLURRY_FLOW_LINE_A','SLURRY_FLOW_LINE_B','SLURRY_FLOW_LINE_C','WAFER_ROTATION','STAGE_ROTATION','HEAD_ROTATION','DRESSING_WATER_STATUS','EDGE_AIR_BAG_PRESSURE'])
output_df3 = pd.DataFrame(repeated_output_values3, columns=['WAFER_ID','STAGE','AVG_REMOVAL_RATE'])

# Merge input and output DataFrames based on ID column
df3 = pd.merge(input_df3, output_df3, on=['WAFER_ID','STAGE'])


# Optionally, assign an index to each row
df3.index = range(1, len(df3) + 1)


# In[56]:


df1.shape


# In[57]:


df2.shape


# In[58]:


df3.shape


# In[59]:


df1=df1.set_index('WAFER_ID')
df2=df2.set_index('WAFER_ID')
df3=df3.set_index('WAFER_ID')
df1=df1.drop(1834206972)
df1=df1.drop(1834206944)
df1=df1.drop(1834206730)
df1=df1.drop(2058207580)
df1['STAGE'] =df1['STAGE'].map({'A': 0, 'B': 1})
df2['STAGE'] =df2['STAGE'].map({'A': 0, 'B': 1})
df3['STAGE'] =df3['STAGE'].map({'A': 0, 'B': 1})
df1 = df1.drop(columns=['MACHINE_ID', 'MACHINE_DATA', 'TIMESTAMP'])
df2 = df2.drop(columns=['MACHINE_ID', 'MACHINE_DATA', 'TIMESTAMP'])
df3 = df3.drop(columns=['MACHINE_ID', 'MACHINE_DATA', 'TIMESTAMP'])


# In[60]:


df1.shape


# In[61]:


df2.head()


# In[62]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df1 = df1.sample(frac=1, random_state=42).reset_index(drop=True)
df2 = df2.sample(frac=1, random_state=42).reset_index(drop=True)
df3 = df3.sample(frac=1, random_state=42).reset_index(drop=True)


# # Define the target columns based on zero-indexing
target_columns = [7, 8, 21, 9, 10, 16, 18]  # Adjusted for zero indexing

# # Select the target columns
y1 = df1.iloc[:, target_columns]
y2 = df2.iloc[:, target_columns]
y3 = df3.iloc[:, target_columns]

# # Select the remaining columns as features
X1 = df1.drop(df1.columns[target_columns], axis=1)
X2 = df2.drop(df2.columns[target_columns], axis=1)
X3 = df3.drop(df3.columns[target_columns], axis=1)

# # Initialize the scaler for the inputs with the range [-1, 1]
scaler_x = MinMaxScaler(feature_range=(-1, 1))
# # Fit the scaler on the entire features and transform them
X_train_scaled = scaler_x.fit_transform(X1)
X_val_scaled = scaler_x.transform(X2)
X_test_scaled = scaler_x.transform(X3)

# # Initialize the scaler for the outputs with the range [0, 1]
scaler_y = MinMaxScaler(feature_range=(0, 1))
# # Fit the scaler on the entire target and transform it
y_train_scaled = scaler_y.fit_transform(y1)
y_val_scaled = scaler_y.transform(y2)
y_test_scaled = scaler_y.transform(y3)


# # Now split the data into training, validation, and test sets
# # Note: Since the indices are specific and non-overlapping, we don't need to shuffle the data

# # Training set
#X_train_scaled = X_scaled[:503683]
#y_train_scaled = y_scaled[:503683]

# # Validation set
#X_val_scaled = X_scaled[503683:588225]
#y_val_scaled = y_scaled[503683:588225]

# # Test set
#X_test_scaled = X_scaled[588225:]
#y_test_scaled = y_scaled[588225:]

# # Now X_train, y_train, X_val, y_val, X_test, and y_test are all scaled and split according to your specifications


# In[126]:


import tensorflow as tf
from tensorflow.keras import Model, layers, initializers, optimizers
import matplotlib.pyplot as plt

class InvertibleNetwork(Model):
    def __init__(self, in_dim=15, out_dim=7, width=64, depth=5, learning_rate=1e-3):
        super(InvertibleNetwork, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.depth = depth

        # Initialize the layers of the network
        self.hidden_layers = [layers.Dense(self.width, activation='tanh') for _ in range(self.depth)]
        self.output_layer = layers.Dense(self.out_dim,activation='sigmoid')
        
        # Parameter to train
        #self.C = tf.Variable(initial_value=tf.random.normal([1], stddev=0.1), dtype=tf.float32, trainable=True)
        self.C = tf.Variable(initial_value=0.001, dtype=tf.float32, trainable=True)

        
        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output

    def train_step(self, x, y_true):
        with tf.GradientTape() as tape:
            y_pred = self(x)
          # Ensure y_pred is of the same data type as y_true
            y_pred = tf.cast(y_pred, y_true.dtype)
        
        # Calculate the mean squared error loss
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
          # Ensure that the operations below are conducted with consistent data types
            avg_output_0_1_3_4 = tf.reduce_mean(tf.gather(y_pred, [0, 1, 3, 4], axis=1), axis=1)
            term_to_power = tf.pow(avg_output_0_1_3_4, 7.0 / 6.0)
            term_to_power = tf.cast(term_to_power, y_true.dtype)  # Cast term_to_power to match y_true and y_pred
        
            diff_output_5_6 = y_pred[:, 6] - y_pred[:, 5]
            diff_output_5_6 = tf.cast(diff_output_5_6, y_true.dtype)  # Cast diff_output_5_6 to match y_true and y_pred
        
            output_2 = y_pred[:, 2]
            output_2 = tf.cast(output_2, y_true.dtype)  # Cast output_2 to match y_true and y_pred
            C_casted = tf.cast(self.C, y_true.dtype)
        # Calculate the constraint term based on the given formula
            constraint_term = output_2 - C_casted * term_to_power * diff_output_5_6
            constraint_loss = tf.reduce_mean(tf.square(constraint_term))
        
        # The constraint_term can then be squared and added to the MSE loss to form the total loss.
            total_loss = mse_loss + 0.1 * constraint_loss

    # Calculate gradients with respect to trainable variables
        gradients = tape.gradient(total_loss, self.trainable_variables)

    # Update weights and C
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return total_loss



# Create an instance of the network
model = InvertibleNetwork()

# Assume you have the following datasets already created
x_train_tensor = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train_scaled, dtype=tf.float32)
x_val_tensor = tf.convert_to_tensor(X_val_scaled, dtype=tf.float32)
y_val_tensor = tf.convert_to_tensor(y_val_scaled, dtype=tf.float32)

batch_size =7  # or another batch size that you wish to use

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train_scaled)).shuffle(buffer_size=len(X_train_scaled)).batch(batch_size, drop_remainder=True)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_scaled, y_val_scaled)).batch(batch_size, drop_remainder=True)

train_dataset = train_dataset.shuffle(buffer_size=len(X_train_scaled)).batch(batch_size)

train_losses = []
val_losses = []

# Training loop with validation
for epoch in range(150):  # Number of epochs
    # Initialize variables to track the loss for each epoch
    epoch_train_losses = []
    epoch_val_losses = []
    
    # Training
    for x_batch, y_batch in train_dataset:
        train_loss = model.train_step(x_batch, y_batch)
        epoch_train_losses.append(train_loss.numpy())
    
    # Record the average loss for this training epoch
    average_train_loss = np.mean(epoch_train_losses)
    train_losses.append(average_train_loss)
    
    # Validation
    for x_batch, y_batch in val_dataset:
        y_batch = tf.cast(y_batch, dtype=tf.float32)
        y_val_pred = model(x_batch)
        val_loss = tf.reduce_mean(tf.square(y_batch - y_val_pred)).numpy()
        epoch_val_losses.append(val_loss)
    
    # Record the average validation loss for this epoch
    average_val_loss = np.mean(epoch_val_losses)
    val_losses.append(average_val_loss)
    

    # Print the loss information at intervals or at certain epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Training Loss: {average_train_loss}, Validation Loss: {average_val_loss},Trained C: {model.C.numpy()}')


# # Lists to keep track of losses
# train_losses = []
# val_losses = []

# # Training loop with validation
# for epoch in range(150):  # Number of epochs
#     # Training step
#     train_loss = model.train_step(x_train_tensor, y_train_tensor)
#     train_losses.append(train_loss.numpy())

#     # Validation step
#     y_val_pred = model(x_val_tensor)
#     val_loss = tf.reduce_mean(tf.square(y_val_tensor - y_val_pred))
#     val_losses.append(val_loss.numpy())

#     if epoch % 10 == 0:
#         print(f'Epoch {epoch}, Training Loss: {train_loss.numpy()}, Validation Loss: {val_loss.numpy()}, Trained C: {model.C.numpy()}')

#After training is complete, evaluate on the test set
#y_test_pred = model(x_test)
#test_loss = tf.reduce_mean(tf.square(y_test - y_test_pred))
#print(f'Test Loss: {test_loss.numpy()}')

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[127]:


# Assuming you have scaled test data X_test_scaled and y_test_scaled
x_test_tensor = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test_scaled, dtype=tf.float32)

# After training is complete, evaluate on the test set
y_test_pred = model(x_test_tensor)
test_loss = tf.reduce_mean(tf.square(y_test_tensor - y_test_pred))
print(f'Test Loss: {test_loss.numpy()}')


# In[128]:


import matplotlib.pyplot as plt
import tensorflow as tf

# Assume 'model' is your trained InvertibleNetwork instance
# and 'x_train_tensor' & 'y_train_tensor' are your training datasets.

# Predict the outputs for the first 100 training set samples
y_pred_train = model(x_train_tensor[:50])

# Extract the 7th value (at index 6 assuming 0-based indexing) for the first 100 predictions
y_pred_train_7th_value = y_pred_train[:, 2]

# Extract the actual 7th value for the first 100 training set samples
y_train_7th_value = y_train_tensor[:50, 2]

# Convert the tensor to a numpy array for plotting
y_pred_train_7th_value_numpy = y_pred_train_7th_value.numpy()
y_train_7th_value_numpy = y_train_7th_value.numpy()

# Plotting the predicted vs actual values for the 7th output for the first 100 training samples
plt.figure(figsize=(10, 5))
plt.plot(y_pred_train_7th_value_numpy, label='Predicted 7th Value')
plt.plot(y_train_7th_value_numpy, label='Actual 7th Value')
plt.title('Predicted vs Actual 7th Output Comparison for First 100 Training Samples')
plt.xlabel('Sample Index')
plt.ylabel('Output Value')
plt.legend()
plt.show()


# In[129]:


# Predict the first 100 values (ensure x_test_tensor has at least 100 samples)
y_test_pred = model.predict(x_test_tensor[:100])

# Extract the 3rd output (index 2) from the predictions and actual values
predicted_3rd_output = y_test_pred[:, 2]
actual_3rd_output = y_test_tensor[:100, 2]

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(predicted_3rd_output, label='Predicted MRR', marker='o')
plt.plot(actual_3rd_output, label='Actual MRR', marker='x')
plt.title('Comparison of Predicted and Actual MRR for First 100 Samples')
plt.xlabel('Sample Index')
plt.ylabel('Output Value')
plt.legend()
plt.show()


# In[167]:


import numpy as np
import matplotlib.pyplot as plt

# Predict the first 100 values (ensure x_test_tensor has at least 100 samples)
y_test_pred = model.predict(x_test_tensor[:100])

# Extract the 3rd output (index 2) from the predictions and actual values
predicted_3rd_output = y_test_pred[:, 2]
actual_3rd_output = y_test_tensor[:100, 2]

# Convert to NumPy arrays if they are TensorFlow tensors
predicted_3rd_output_np = predicted_3rd_output.numpy() if hasattr(predicted_3rd_output, 'numpy') else predicted_3rd_output
actual_3rd_output_np = actual_3rd_output.numpy() if hasattr(actual_3rd_output, 'numpy') else actual_3rd_output

# Create dummy arrays with the same number of columns as the original dataset
num_features = y_train_scaled.shape[1]  # Assuming the same number of features as in y_train_scaled
dummy_predicted = np.zeros((100, num_features))
dummy_actual = np.zeros((100, num_features))

# Place the predicted and actual values in the 3rd column (index 2)
dummy_predicted[:, 2] = predicted_3rd_output_np
dummy_actual[:, 2] = actual_3rd_output_np

# Inverse transform to get original scale values
predicted_3rd_output_inverted = scaler_y.inverse_transform(dummy_predicted)[:, 2]
actual_3rd_output_inverted = scaler_y.inverse_transform(dummy_actual)[:, 2]

# Plotting the inverted values
plt.figure(figsize=(12, 6))
plt.plot(predicted_3rd_output_inverted, label='MRR Predicted Value', marker='o')
plt.plot(actual_3rd_output_inverted, label='MRR Actual Value', marker='x')
plt.title('Comparison of Predicted and Actual MRR for First 100 Samples')
plt.xlabel('Sample Index')
plt.ylabel('MRR(nm/min)')
plt.legend()
plt.show()


# In[168]:


import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Assuming 'actual_3rd_output_inverted' and 'predicted_3rd_output_inverted' are already defined

# Calculate Pearson's correlation coefficient (R)
r, _ = pearsonr(actual_3rd_output_inverted, predicted_3rd_output_inverted)

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Scatter plot with line of best fit for actual vs predicted
ax.scatter(actual_3rd_output_inverted, predicted_3rd_output_inverted, label='Predicted', alpha=0.5)
ax.plot([min(actual_3rd_output_inverted), max(actual_3rd_output_inverted)],
         [min(actual_3rd_output_inverted), max(actual_3rd_output_inverted)],
         'k--', lw=2, label='Actual')
ax.set_title(f'R = {r:.3f}')
ax.set_xlabel('Ground Truth (nm/min)')
ax.set_ylabel('Predicted Results (nm/min)')
ax.legend()

# Display the plot
plt.show()



# In[151]:


# Predict the scaled target values for the test set
y_test_pred_scaled = model.predict(x_test_tensor)

# Inverse transform the predicted scaled values to the original scale
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

# Inverse transform the actual scaled target values to the original scale
y_test_actual = scaler_y.inverse_transform(y_test_tensor)

# Calculate the MSE for the 3rd output using the rescaled values
mse_3rd_output = tf.reduce_mean(tf.square(y_test_actual[:, 2] - y_test_pred[:, 2]))

# Evaluate the tensor to get the value
mse_3rd_output_value = mse_3rd_output.numpy()
print(f"MSE for the 3rd output on original scale: {mse_3rd_output_value}")


# In[131]:


# Predict the scaled target values for the training set
y_train_pred_scaled = model(x_train_tensor)

# Inverse transform the predicted scaled values to the original scale
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)

# Inverse transform the actual scaled target values to the original scale
y_train_actual = scaler_y.inverse_transform(y_train_tensor)

# Calculate the MSE for the training data using the rescaled values
mse_train_original = tf.reduce_mean(tf.square(y_train_actual[:, 2] - y_train_pred[:, 2]))

# Evaluate the tensor to get the value, if necessary
mse_train_original_value = mse_train_original.numpy()
print(f"MSE for the training data on original scale: {mse_train_original_value}")

