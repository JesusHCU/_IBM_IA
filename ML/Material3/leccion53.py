#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[4]:


diccionario1 = {'A':[1,2,3], 'B':[4,5,6], 'C':[7,8,9]}
dataframe1 = pd.DataFrame(diccionario1)


# In[5]:


diccionario2 = {'A':[11,12,13], 'B':[14,15,16], 'C':[17,18,19]}
dataframe2 = pd.DataFrame(diccionario2)


# In[9]:


pd.concat([dataframe1,dataframe2])


# In[ ]:




