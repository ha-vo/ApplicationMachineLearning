a = 5
b = 3
if a > b:
    a = a*2 + 3
    b = b - 6
    c = a/b
    print(c)
c = a + b + \
    10 * a - b/4 - \
    5 + a* 3
print(c)

a = 5
b = 3
if a>b:
    print("True")
    print(a)
else:
    print("False")
    print(b)
    
a = 1
b = 10
while a < b :
    a += 1
    print(a)
    
for i in range(1,10):
    print(i)
    
def binhphuong(number):
    return number*number
print(binhphuong(5))

a = 5
b = -1
c = 1.234

str1 = "Hello"
str2 = "welcome"
str3 = "abcdef12345"

cats = ['Tom','Snappy','Kitty','Jessie','Chester']
print(cats[2])
print(cats[-1])
print(cats[1:3])
print(cats)
cats.append('Jerry')
print(cats)
cats[-1] = 'Jerrt Cat'
print(cats)
del cats[1]
print(cats)

import numpy as np

a = np.array([0,1,2,3,4,5])
print(a)
print(a.ndim)
print(a.shape)
print(a[a>3])
a[a>3] = 10
b = a.reshape((3,2))
print(b)
print(b[2][1])
b[2][0] = 50
c = b*2
print(c)

import pandas as pd
dt =pd.read_csv("play_tennis.csv",delimiter=',')
print(dt.head())
print(dt.tail(7))

print(dt.loc[3:8])
print(dt.iloc[:,3:6])
print(dt.iloc[5:10,3:4])
print(dt.outlook)


