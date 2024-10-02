#!/usr/bin/env python
# coding: utf-8

# # İLERİ SEVİYE MÜŞTERİ SEGMENTASYONU PROJESİ
# 
# Bu projemizde Massachusetts Institute of Technology (MIT) tarafından geliştirilmiş ileri seviye bir kütüphane kullanacağız.
# Verilerimiz komplex olduğu için burada K-Means kullanamıyoruz.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
# kmodes kütüphanesi standart distribution içinde gelmez sizin install etmeniz gerekebilir.
# Massachusetts Institute of Technology (MIT) tarafından geliştirilmiş ileri seviye bir kütüphanedir.
# https://anaconda.org/conda-forge/kmodes linkinde detay mevcuttur
# conda install -c conda-forge kmodes   komutu ile Anaconda üzerinde kurabilirsiniz..
from kmodes.kprototypes import KPrototypes  
from kmodes.kmodes import KModes


# In[3]:


df = pd.read_csv("segmentation_data.csv")
df.head()


# In[4]:


df.tail()


# In[5]:


# We have no null data see:
df.isnull().sum()


# In[6]:


# ## Income ve Age Data Normalization

# Before Scaling/Normalization we keep our normal values in temp variables..
df_temp = df[['ID','Age', 'Income']]
df_temp


# In[7]:


scaler = MinMaxScaler()

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])

scaler.fit(df[['Income']])
df['Income'] = scaler.transform(df[['Income']])


# In[8]:


# Drop ID before analysis..
# Since ID is not used in analysis...
df = df.drop(['ID'], axis=1)


# In[9]:



mark_array= df.values

mark_array[:, 2] = mark_array[:, 2].astype(float)
mark_array[:, 4] = mark_array[:, 4].astype(float)


# In[10]:


df.head()


# In[11]:


# Build our model...

kproto = KPrototypes(n_clusters=10, verbose=2, max_iter=20)
clusters = kproto.fit_predict(mark_array, categorical=[0, 1, 3, 5, 6])


print(kproto.cluster_centroids_)

len(kproto.cluster_centroids_)


# In[12]:


cluster_dict=[]
for c in clusters:
    cluster_dict.append(c)



df['cluster']=cluster_dict



# Put original columns from temp to df:
df[['ID','Age', 'Income']] = df_temp


# In[13]:


df[df['cluster']== 0].head(10)


# In[14]:



df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
df4 = df[df.cluster==3]
df5 = df[df.cluster==4]
df6 = df[df.cluster==6]
df7 = df[df.cluster==7]
df8 = df[df.cluster==8]
df9 = df[df.cluster==9]
df10 = df[df.cluster==10]


plt.figure(figsize=(15,15))
plt.xlabel('Age')
plt.ylabel('Income')

plt.scatter(df1.Age, df1['Income'],color='green', alpha = 0.4)
plt.scatter(df2.Age, df2['Income'],color='red', alpha = 0.4)
plt.scatter(df3.Age, df3['Income'],color='gray', alpha = 0.4)
plt.scatter(df4.Age, df4['Income'],color='orange', alpha = 0.4)
plt.scatter(df5.Age, df5['Income'],color='yellow', alpha = 0.4)
plt.scatter(df6.Age, df6['Income'],color='cyan', alpha = 0.4)
plt.scatter(df7.Age, df7['Income'],color='magenta', alpha = 0.4)
plt.scatter(df8.Age, df8['Income'],color='gray', alpha = 0.4)
plt.scatter(df9.Age, df9['Income'],color='purple', alpha = 0.4)
plt.scatter(df10.Age, df10['Income'],color='blue', alpha = 0.4)


# kmeans_modelim.cluster_centers_ numpy 2 boyutlu array olduğu için x ve y sütunlarını kmeans_modelim.cluster_centers_[:,0] 
# ve kmeans_modelim.cluster_centers_[:,1] şeklinde scatter plot için alıyoruz:
#plt.scatter(kmeans_modelim.cluster_centers_[:,0], kmeans_modelim.cluster_centers_[:,1], color='blue', marker='X', label='centroid')
plt.legend()
plt.show()


# In[ ]:




