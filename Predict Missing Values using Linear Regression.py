#!/usr/bin/env python
# coding: utf-8

# # Linear Regression for missing data prediction
# Insulin has the highest no of missing values. Therefore, glucose and output will act as the independent variables for predicting insulin (dependent variable).

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for visualization and plot
import csv,sklearn

from subprocess import check_output


# In[2]:


#read newly coded diabetes data 
data = pd.read_csv('input/diabetes2.csv',encoding='latin1')
data.head()


# In[3]:


#here only variables which highly correlate with outcome from the second notebook is retained
data = data[['Glucose','Insulin','Output']]
data.head()


# In[4]:


#delete zero row entries
data= data[data['Insulin'] != 0]
data= data[data['Glucose'] != 0]

#save non-zero entry data as 
data.to_csv('input/diabetesNonzero.csv', index=False)

#perform descriptive analysis
data.describe()


# In[5]:


from sklearn.model_selection import train_test_split
splitRatio = 0.2

train , test = train_test_split(data,test_size = splitRatio,random_state = 123)

X_train = train[[x for x in train.columns if x not in ["Insulin"]]]
y_train = train[["Insulin"]]

X_test  = test[[x for x in test.columns if x not in ["Insulin"]]]
y_test  = test[["Insulin"]]


# In[6]:


from sklearn.model_selection import train_test_split 
#X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[64]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

#model = Ridge()
#model = Lasso()
#best model
model =  LinearRegression()
model.fit(X_train,y_train)
prediction = model.predict(X_test)


# In[65]:


#visualize train and test accuracy
#from sklearn.metrics import score

#print("Train acc: " , model.score(X_train, y_train))
#print("Test acc: ", model.score(X_test, y_test))

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, prediction))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, prediction))


# In[66]:


#Test model on unknown sample, given glucose and output label
new_df = pd.DataFrame([[144,2]])

# We predict insulin
prediction = model.predict(new_df)


print(prediction.astype(int))


# In[67]:


#new_df = pd.DataFrame([[141,44]])
new_df = pd.read_csv('input/diabetestest2.csv',encoding='latin1')

# We predict the outcome
prediction = model.predict(new_df)
prediction = prediction.astype(int)


# In[52]:


out=pd.DataFrame(prediction, columns=['Insulin'])
out.to_csv('input/diabetespredresult.csv',index=False, header=True)


# In[ ]:




