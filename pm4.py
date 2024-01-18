# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 21:02:25 2023

@author: Manohar
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from google.colab import drive
#importing csv file 
#drive.mount('/content/drive')
#path = r"/content/drive/MyDrive/ml_asgn1/Dataset1.csv"
path=r"D:\rough\Dataset1.csv"
df = pd.read_csv(path)
print(df.head())

print(df.shape)

#%%


#PM4
# shuffle feature columns while retaining the original feature names
num_data, num_columns = df.shape
columns_to_shuffle = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']
shuffled_columns = df[columns_to_shuffle].sample(frac=1, axis=1)
#shuffled_columns = shuffled_columns.reindex(columns=columns_to_shuffle)
#shuffled_columns
df[columns_to_shuffle] = shuffled_columns.values
df

# Split dataset into train and test data
train_data = df.sample(frac=0.67)
test_data = df.drop(train_data.index)

X = train_data.iloc[:, 2:].values
Y = train_data.iloc[:, 1].values
mean1 = np.nanmean(X)
X[np.isnan(X)] = mean1
y = np.array([1 if i == 'M' else -1 for i in Y])
w=np.zeros(X.shape[1])
#print(X.shape[0])
epochs = 100

for i in range(epochs):
    for j in range(X.shape[0]):
      m=np.dot(X[j],w)
      if m * y[j] <= 0:
                w += X[j] * y[j]
                #print('hi')
                # print(w)
# Test perceptron model
X_test = test_data.iloc[:, 2:].values
Y_test = test_data.iloc[:, 1].values
y_test = np.array([1 if i == 'M' else -1 for i in Y_test])
y_pred = np.sign(np.dot(X_test, w))
#print(y_pred)
# Calculate accuracy
accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
print('epochs',epochs)
print("Accuracy is ", accuracy*100,"%")