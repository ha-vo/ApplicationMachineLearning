#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB


# câu a

# In[3]:


winequality = pd.read_csv('D:\winequality-red.csv',sep=';')


# câu b

# In[4]:


len(winequality)  # 1599 phan tu


# In[5]:


np.unique(winequality.quality) # 3 4 5 6 7 8


# In[6]:


winequality.quality.value_counts()


# 5    681
# 6    638
# 7    199
# 4     53
# 8     18
# 3     10

# câu c

# In[7]:


dulieu = winequality.iloc[:,:11]
nhan = winequality.quality


# In[26]:


x_train,x_test, y_train, y_test = train_test_split(dulieu,nhan,test_size=0.4,random_state = 0)


# In[27]:


len(x_test)


# 640

# In[28]:


test = np.unique(y_test)
len(test)


# 6

# câu d

# In[29]:


mohinh_knn = KNeighborsClassifier(n_neighbors = 7)
mohinh_knn.fit(x_train,y_train)
y_pred = mohinh_knn.predict(x_test)


# 1) Độ chính xác độ chính xác tổng thể và độ chính xác của từng lớp cho toàn bộ dữ
# liệu trong tập test
# * Độ chính xác độ chính xác tổng thể

# In[30]:


doCX = accuracy_score(y_test,y_pred) * 100
doCX


# -> Độ chính xác tổng thể là: 50.9375

# * Độ chính xác của từng lớp

# In[19]:


cnf_matrix = confusion_matrix(y_test,y_pred)
cnf_matrix


# [[  0,   0,   0,   3,   1,   0],
#  [  0,   1,  14,   7,   1,   0],
#  [  0,   0, 169, 103,   7,   0],
#  [  0,   0, 108, 139,  21,   0],
#  [  0,   0,  16,  27,  17,   0],
#  [  0,   0,   2,   1,   3,   0]

# 2) Độ chính xác độ chính xác tổng thể và độ chính xác của từng lớp cho 8 phần tử đầu tiên của tập test
# * Độ chính xác độ chính xác tổng thể

# In[34]:


x_test8 = x_test.iloc[:8,]
y_test8 = y_test.iloc[:8,]


# In[43]:


y_pred8 = mohinh_knn.predict(x_test8)
doCX8 = accuracy_score(y_test8, y_pred8) * 100
doCX8


# => Độ chính xác độ chính xác tổng thể là: 50.0

# * độ chính xác của từng lớp

# In[37]:


cnf_matrix8 = confusion_matrix(y_test8,y_pred8)
cnf_matrix8


# [[0, 3, 0],
#        [1, 3, 0],
#        [0, 0, 1]]

# câu e) Bayes ngây thơ

# In[39]:


model = GaussianNB()


# In[40]:


model.fit(x_train,y_train)


# In[41]:


y_predNB = model.predict(x_test)


# * độ chính xác tổng thể

# In[44]:


doCXNB = accuracy_score(y_test,y_predNB) * 100
doCXNB


# 53.90625

# * Độ chính xác từng lớp

# In[46]:


cnf_matrixNB = confusion_matrix(y_test,y_predNB)
cnf_matrixNB


# [[  0,   0,   3,   1,   0,   0],
#        [  3,   0,   9,  10,   0,   1],
#        [  4,   8, 180,  68,  19,   0],
#        [  2,   2,  64, 129,  66,   5],
#        [  1,   0,   1,  19,  35,   4],
#        [  0,   0,   0,   2,   3,   1]]

# câu f) Với nghi thức hold-out (2/3 để học, 1/3 để kiểm tra), so sánh độ chính xác tổng thể
# của mô hình KNN và Bayes thơ ngây

# In[48]:


xTrain,xTest,yTrain,yTest = train_test_split(dulieu,nhan,test_size = 1/3, random_state = 0)


# In[49]:


model_knn = KNeighborsClassifier(n_neighbors = 7)
model_knn.fit(xTrain,yTrain)
yPred = mohinh_knn.predict(xTest)


# In[51]:


dCXKNN = accuracy_score(yTest,yPred) * 100
dCXKNN
#51.41

# Ta thấy độ chính xác của mô hình knn = 51.41 nhỏ hơn độ chính xác của Bayes = 53.91
