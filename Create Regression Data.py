#!/usr/bin/env python
# coding: utf-8

# # Goals:
# #### -Re-label data to include three classes: Normal, Prediabetes, and Diabetes 
# #### -Select data entries for learning "relationship between glucose, insulin and outcome"

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for visualization and plot
import csv,sklearn

from subprocess import check_output


# In[2]:


data = pd.read_csv('input/diabetes.csv',encoding='latin1')
data.head()


# In[3]:


#Data labelling!
#This is done to elimiate the inconsistencies in the original data. For example, a person is said to have diabetes but the glucose value is 0
#Define a new column 'Output' for reassining label to data based on medical standard. Glucose >125 is known as diabetes,
#Glucose 125 to 99 is prediabetes, while Glucose 99 <70 is for normal patients, and <70 is diabetes

data['Output'] = data['Glucose'].apply(lambda x: 'diabetes' if x > 125 else 'prediabetes' if x > 99 and x <= 125 else 'normal' if x > 70 else 'diabetes')
data.head()


# In[4]:


#Code data to have numeric labels. The labels are normal=0, prediabetes=1, diabetes=2
data['Output'] = data['Output'].replace(['normal','prediabetes', 'diabetes'],[0,1,2])
data.head()

#save data
data.to_csv('input/diabetes2.csv',index=False, header=True)


# In[5]:


#delete zero row within the insulin column entries
data= data[data['Insulin'] == 0]

#perform descriptive analysis
data.describe()


# In[6]:


#Select only column data of interest
data = data[['Glucose','Output']]
data.head()


# In[7]:


#we can afford to remove 2 rows of glucose with zero entries
data= data[data['Glucose'] != 0]
data.describe()


# In[8]:


data.to_csv('input/diabetestest2.csv',index=False, header=True)

