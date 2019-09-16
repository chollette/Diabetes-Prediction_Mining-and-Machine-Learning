#!/usr/bin/env python
# coding: utf-8

# # Analysis and Visualization
# 
# ### The On-Set of Diabetes Prediction using PIMA INDIANs Dataset

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


#read diabetes data 
data = pd.read_csv('input/diabetes.csv',encoding='latin1')
data.head()


# In[3]:


#Statistically find relationship between variables using correlation
#and visualize the correlations using a heatmap.
VarCorr = data.corr()
print(VarCorr)
sns.heatmap(VarCorr,xticklabels=VarCorr.columns,yticklabels=VarCorr.columns)


# ## Interpretation 
# 
# The heatmap indicates that the brighter the colors the higher the correlation and vice versa.
# 
# We can see that glucose is highly correlated to the dependent varaiable, which invariables means that the above medical facts is shown to be true in the given data.
# 
# Also, Insulin is the next correlated independent variable in the given data, but  obviously does not correlate with outcome which medically is true because Insulin levels are used to predict the type of diabetes.
# 
# Based on these facts, the following conclusion is made:
# 
# That Glucose is the major predictor of diabetes
# Insulin is the major indicator of the type of daibetes
# 
# For the above reasons, only the glucose and insulin will be used to predict diabtes and type of diabetes.
