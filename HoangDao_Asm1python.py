#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np


# In[19]:


df=pd.read_csv('imports-85.csv')


# In[20]:


df


# In[21]:


df=df.replace("?",np.NaN)


# In[22]:


#Transforming to correct type of data for processing
df['normalized-losses']=df['normalized-losses'].astype(str).astype(float)
df['doors']=df['doors'].astype(str)
df['bore']=df['bore'].astype(str).astype(float)
df['stroke']=df['stroke'].astype(str).astype(float)
df['hp']=df['hp'].astype(str).astype(float)
df['peak-rpm']=df['peak-rpm'].astype(str).astype(float)
df['price']=df['price'].astype(str).astype(float)


# In[23]:


#Observing the mean of continuous attributes
df.mean()


# In[24]:


#Observing the median of continuous attributes
df.median()


# In[27]:


#Using mean to fill missing values in normalized-losses, bore,stroke columns
df['normalized-losses'].fillna(value=df['normalized-losses'].mean(), inplace=True)
df['bore'].fillna(value=df['bore'].mean(), inplace=True)
df['stroke'].fillna(value=df['stroke'].mean(), inplace=True)


# In[28]:


#Using median to fill missing values in price, hp, peak-rpm columns
df['price'].fillna(value=df['price'].median(), inplace=True)
df['hp'].fillna(value=df['hp'].median(), inplace=True)
df['peak-rpm'].fillna(value=df['peak-rpm'].median(), inplace=True)


# In[29]:


#Observing attribute doors
df.loc[(df['body']=='sedan') & (df['doors']=='two')]


# In[30]:


df.loc[(df['body']=='hatchback')&(df['doors']=='two')]


# In[31]:


df.loc[(df['body']=='hatchback')&(df['doors']=='four')]


# In[32]:


df.loc[df['body']=='hardtop']


# In[33]:


df.loc[df['doors']=='nan']


# In[34]:


#After observing, I come up with the decision tree model to fill in doors attribute (in the report).


# In[35]:


#I take the mode of sedan body.
df.loc[(df['body']=='sedan')].mode()


# In[36]:


#Therefore, I fill missing values in doors attribute with "four" value
df['doors']=df['doors'].replace('nan','four')


# In[40]:


#Observing the data frame again
df.head(60)


# In[ ]:




