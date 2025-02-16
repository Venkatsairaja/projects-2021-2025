#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import warnings

warnings.filterwarnings('ignore')


# In[3]:


data = 'country-wise-average.csv'

df = pd.read_csv(data)


# In[4]:


df.head()


# In[5]:


X = df

y = df['Country']


# In[6]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X['Country'] = le.fit_transform(X['Country'])

y = le.transform(y)


# In[7]:


X.info()


# In[8]:


X.head()


# In[9]:


cols = X.columns


# In[10]:


from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

X = ms.fit_transform(X)


# In[11]:


X = pd.DataFrame(X, columns=[cols])


# In[12]:


X.head()


# In[16]:


from sklearn.cluster import KMeans
import numpy as np

# Generate some sample data
X = np.random.rand(100, 2)

# Instantiate KMeans with desired number of clusters
kmeans = KMeans(n_clusters=3)

# Fit the model to the data
kmeans.fit(X)

# Access the cluster centers
cluster_centers = kmeans.cluster_centers_


# In[17]:


kmeans.inertia_


# In[ ]:





# In[21]:


from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()


# In[23]:


from sklearn.cluster import KMeans
import numpy as np

# Generate some sample data
X = np.random.rand(100, 2)

# Instantiate KMeans with desired number of clusters
kmeans = KMeans(n_clusters=3)

# Fit the model to the data and obtain cluster labels
kmeans.fit(X)
labels = kmeans.labels_

# Assuming `y` represents true labels
y = np.random.randint(0, 3, size=100)  # Example of generating random true labels

# Check how many of the samples were correctly labeled
correct_labels = sum(y == labels)

# Print the result
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))


# In[ ]:




