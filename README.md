# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries.
2. the dataset/analyze the data.
3. Preprocess  the data.
4. Split the x,y Training/Testing data.
5. Import model to fit.
6. Use the model to predict on the test data.
7. Evaluate the metrices.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: 
RegisterNumber:  
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
```
```
data = pd.read_csv("/content/Mall_Customers.csv")
data.head()
```
![alt text](<Screenshot 2024-11-04 163043.png>)
```
data.info()
```
![alt text](<Screenshot 2024-11-04 163123.png>)
```
data.isnull().sum()
```
![alt text](<Screenshot 2024-11-04 163203.png>)
```
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("No.of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
```
![alt text](<Screenshot 2024-11-04 163238.png>)
```
km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])
KMeans(n_clusters=5)
y_pred=km.predict(data.iloc[:,3:])
y_pred
```
![alt text](<Screenshot 2024-11-04 163308.png>)
```
data["cluster"]=y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
```
```
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"], color = "gold")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"], color = "pink")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"], color = "green")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"], color = "blue")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"], color = "red")
plt.show()
```
![alt text](<Screenshot 2024-11-04 163352.png>)




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
