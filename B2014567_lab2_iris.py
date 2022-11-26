from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np


iris_dt = load_iris()


x_train,x_test, y_train,y_test = train_test_split(iris_dt.data,iris_dt.target,test_size = 1/3.0, random_state = 5)

print("tap du lieu dc train")
print(x_test,x_train,sep='\n')
print(y_test,y_train,sep = '\n')

print(iris_dt.data[1:5])
print(iris_dt.target[1:5])

mohinh_knn = KNeighborsClassifier(n_neighbors = 5)
mohinh_knn.fit(x_train,y_train)
y_pred = mohinh_knn.predict(x_test)
print(y_test)
print(mohinh_knn.predict([[4,4,3,3]]))

print("Accuracy is ", accuracy_score(y_test,y_pred) * 100)

print(confusion_matrix(y_test,y_pred, labels = [2,0,1]))

dulieu = pd.read_csv('iris_data.csv')
x = dulieu.iloc[0:,0:4]
y = dulieu.nhan
print("shape ofx: ",x.shape,"shape of y:",y.shape)

print(x)
print(y)

x_train,x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)
model = GaussianNB()
model.fit(x_train,y_train)
print(model)

thucte = y_test
dubao = model.predict(x_test)

print(thucte)
print(dubao)

cnf_matrix_gnb = confusion_matrix(thucte,dubao)
print(cnf_matrix_gnb)





