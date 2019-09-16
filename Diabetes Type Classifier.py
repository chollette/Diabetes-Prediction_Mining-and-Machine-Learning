#!/usr/bin/env python
# coding: utf-8

# # Actual Classification:
# ## Goal: determine the type of diabetes;  type-1 or type-2!

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for visualization and plot
import csv,sklearn

from subprocess import check_output


# In[2]:


data = pd.read_csv('input/diabetesmerge.csv',encoding='latin1')
data.head()


# ### Re-label

# In[3]:


#where Output == 2 change to 2 if Insulin < 30 elif change to 3 if insulin > 0
def myfunc(x,y):
    if x <= 30 and y == 2:
        return y
    elif x > 30 and y == 2:
        return y + 1
    else:
        return y
    
data['Output'] = data.apply(lambda x: myfunc(x.Insulin, x.Output), axis=1)


# In[4]:


#Another way:
#mask = (data['Output'] == 2) & (data['Insulin'] >30)
#data['Output'][mask] = 3
#data


# In[5]:


data.to_csv('input/diabetestype.csv', index=False)
data = pd.read_csv('input/diabetestype.csv',encoding='latin1')


# In[6]:


#data = pd.read_csv('input/diabetesgi.csv',encoding='latin1')
data.head()


# In[7]:


data.describe()


# In[8]:


from sklearn.model_selection import train_test_split
splitRatio = 0.2

train , test = train_test_split(data,test_size = splitRatio,random_state = 123)

X_train = train[[x for x in train.columns if x not in ["Output"]]]
y_train = train[["Output"]]

X_test  = test[[x for x in test.columns if x not in ["Output"]]]
y_test  = test[["Output"]]


# In[9]:


from sklearn.model_selection import train_test_split 
#X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[11]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

model = KNeighborsClassifier()
#model = LogisticRegression()
#model = GaussianNB()
#model =  LinearRegression()

#X_train = X_train.reshape(-1,1)
#X_test = X_test.reshape(-1,1)

model.fit(X_train,y_train)
prediction = model.predict(X_test)
accuracy_score(y_test,prediction)


# In[12]:


print("Train acc: " , model.score(X_train, y_train))
print("Test acc: ", model.score(X_test, y_test))


# In[13]:


new_df = pd.DataFrame([[160,30]])
#new_df = pd.read_csv('input/diabetestest.csv',encoding='latin1')

# We predict the outcome
prediction = model.predict(new_df)

if prediction == 2:
    print('Type 1 diabetes')
elif prediction == 3:
    print('Type 2 diabetes')


# In[ ]:




