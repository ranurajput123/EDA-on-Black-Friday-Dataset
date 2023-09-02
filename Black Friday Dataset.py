#!/usr/bin/env python
# coding: utf-8

# Black Friday Dataset
# -

# Problem Statement :
# - A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month. The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.
# 
# - Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products.

# 1). Cleaning and Prepring the data for model training

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# train dataset
df_train = pd.read_csv('train.csv')


# In[3]:


df_train.head()


# In[4]:


# test dataset
df_test = pd.read_csv('test.csv')


# In[5]:


df_test.head()


# In[6]:


## combine train and test dataset
df = df_train.append(df_test)


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.drop(['User_ID'],axis=1,inplace=True)


# In[11]:


df.head()


# 2). Handling Categorical Features

# 2.1). Gender

# In[13]:


## gender column : F--> 0 and M--> 1
df['Gender'] = df['Gender'].map({'F':0,'M':1})


# In[14]:


df.head()


# 2.2). Age

# In[15]:


df['Age'].unique()


# In[16]:


df['Age'] = df['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})


# In[17]:


df.head()


# 2.3). City Category

# In[18]:


df_city = pd.get_dummies(df['City_Category'],drop_first=True)


# In[19]:


df_city.head()


# In[22]:


df = pd.concat([df,df_city],axis=1)


# In[23]:


df.head()


# In[25]:


## drop city category
df.drop('City_Category',axis=1,inplace=True)


# In[26]:


df.head()


# In[27]:


## missing values
df.isnull().sum()


# In[29]:


## Product_Category_2
df['Product_Category_2'].unique()


# In[30]:


df['Product_Category_2'].value_counts()


# In[37]:


df['Product_Category_2'].mode()[0]


# In[38]:


## replace the missing value with mode
df['Product_Category_2'] = df['Product_Category_2'].fillna(df['Product_Category_2']).mode()[0]


# In[40]:


df['Product_Category_2'].isnull().sum()


# In[41]:


## Product_Category_3
df['Product_Category_3'].unique()


# In[42]:


df['Product_Category_3'].value_counts()


# In[43]:


df['Product_Category_3'].mode()[0]


# In[44]:


## replace the missing value with mode
df['Product_Category_3'] = df['Product_Category_3'].fillna(df['Product_Category_3']).mode()[0]


# In[45]:


df['Product_Category_2'].isnull().sum()


# In[46]:


df.columns


# In[47]:


df['Stay_In_Current_City_Years'].unique()


# In[50]:


df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].str.replace('+','')


# In[51]:


df.head()


# In[52]:


df.info()


# - 'Stay_In_Current_City_Years' is object.
# - Convert object into integer.

# In[53]:


df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].astype(int)


# In[54]:


df.info()


# - B and C ---> uint8

# In[56]:


df['B'] = df['B'].astype(int)
df['C'] = df['C'].astype(int)


# In[57]:


df.info()


# Visualization
# -

# In[59]:


sns.barplot('Age','Purchase',hue='Gender',data=df)


# Observation :
# - Purchasing of men is high than women.
# - Purchasing of all age group is almost simialr.

# In[60]:


sns.barplot('Occupation','Purchase',hue='Gender',data=df)


# In[62]:


sns.barplot('Product_Category_1','Purchase',hue='Gender',data=df)


# In[65]:


sns.barplot('Product_Category_2','Purchase',hue='Gender',data=df)


# In[64]:


sns.barplot('Product_Category_3','Purchase',hue='Gender',data=df)


# In[66]:


df.head()


# Feature Scaling
# -

# In[68]:


df_test = df[df['Purchase'].isnull()]


# In[69]:


df_train = df[~df['Purchase'].isnull()]


# In[83]:


X = df_train.drop('Purchase',axis=1)


# In[84]:


X.head()


# In[85]:


y = df_train['Purchase']


# In[86]:


y

