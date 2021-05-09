#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from sklearn.datasets import load_breast_cancer


# In[10]:


cancer=load_breast_cancer()


# In[14]:


cancer.keys()


# In[17]:


df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[18]:


df.head()


# In[19]:


from sklearn.preprocessing import StandardScaler


# In[22]:


scaler=StandardScaler()


# In[23]:


scaler.fit(df)


# In[26]:


dfscale=scaler.transform(df)


# In[27]:


from sklearn.decomposition import PCA


# In[29]:


pca=PCA(n_components=2)#converting 30 dimension data into two dimensions 


# In[30]:


pca.fit(dfscale)


# In[32]:


xpca=pca.transform(dfscale)


# In[36]:


plt.figure(figsize=(8,6))
plt.scatter(xpca[:,0],xpca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('first principle component')
plt.ylabel('seconf principle component')


# In[ ]:




