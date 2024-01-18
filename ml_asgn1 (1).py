import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from google.colab import drive
#importing csv file 
#drive.mount('/content/drive')
#path = r"/content/drive/MyDrive/ml_asgn1/Dataset1.csv"
path=r"D:\rough\Dataset1.csv"
df = pd.read_csv(path)
#print(df.head())

#print(df.shape)
#%%

#PM1

# Split dataset into train and test data
train_data = df.sample(frac=0.67)
test_data = df.drop(train_data.index)
pm1_accuracies = []
pm1_precisions = []
pm1_recalls = []
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
tp1 = np.sum((y_test == 1) & (y_pred == 1))
tn1 = np.sum((y_test == -1) & (y_pred == -1))
fp1 = np.sum((y_test == -1) & (y_pred == 1))
fn1 = np.sum((y_test == 1) & (y_pred == -1))
accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
precision1 = tp1 / (tp1 + fp1)
recall1 = tp1 / (tp1 + fn1)
print('epochs',epochs)
print("Accuracy is ", accuracy*100,"%")
print(precision1*100)
print(recall1*100)
#%%
#PM2
# Split dataset into train and test data
train_data2 = train_data.sample(frac=1)
# test_data = df.drop(train_data.index)
X2 = train_data.iloc[:, 2:].values
Y2 = train_data.iloc[:, 1].values
mean2 = np.nanmean(X2)
X2[np.isnan(X2)] = mean2
y2 = np.array([1 if i == 'M' else -1 for i in Y])
w2=np.zeros(X2.shape[1])
#print(w)
epochs2 = 100

for i in range(epochs2):
    for j in range(X2.shape[0]):
      m2=np.dot(X2[j],w2)
      if m2 * y2[j] <= 0:
                w2 += X2[j] * y2[j]
                # print(w)



# Test perceptron model
#print(w)
X_test2 = test_data.iloc[:, 2:].values
Y_test2 = test_data.iloc[:, 1].values
y_test2 = np.array([1 if i == 'M' else -1 for i in Y_test2])
y_pred2 = np.sign(np.dot(X_test2, w2))
#print(y_pred2)
# Calculate accuracy
accuracy2 = np.sum(y_pred2 == y_test2) / y_test2.shape[0]
print('epochs2',epochs2)
print("Accuracy is ", accuracy2*100,"%")


#%%

#PM3
pm1_accuracies = []
pm1_precisions = []
pm1_recalls = []
tp1=0
fp1=0
fn1=0
tn1=0
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
w3 = np.zeros(X_norm.shape[1])
epochs3 = 100
#print('mean ',X.mean(axis=0))
#print('std ',X.std(axis=0))
#print(X_norm)
for i in range(epochs3):
    for j in range(X_norm.shape[0]):
        #print('X_norm[j]',(X_norm[j]))
        #print('w3',w3)
        m3=np.dot(X_norm[j],w3)
        #print('m3',m3)
        if m3*y[j] <= 0:
            #print('hi')
            w3 += X_norm[j] * y[j]


# Test perceptron model
X_test_norm = (X_test - X.mean(axis=0)) / X.std(axis=0)
correct_norm = 0
y_pred3 = np.sign(np.dot(X_test_norm, w3))

# Calculate accuracy
tp3 = np.sum((y_test == 1) & (y_pred3 == 1))
tn3 = np.sum((y_test == -1) & (y_pred3 == -1))
fp3 = np.sum((y_test == -1) & (y_pred3 == 1))
fn3 = np.sum((y_test == 1) & (y_pred3 == -1))

precision3 = tp3 / (tp3 + fp3)
recall3 = tp3 / (tp3 + fn3)
accuracy3 = np.sum(y_pred3 == y_test) / y_test.shape[0]
print('epochs3',epochs3)
print("Accuracy is ", accuracy3*100,"%")
print(precision3*100)
print(recall3*100)



