import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as KM


file = pd.read_csv("Mall_Customers.csv")
X = file.iloc[:, 3:5].values

elbow = []
for i in range(10):
	km = KM(n_clusters = i+1)
	km.fit(X)
	elbow.append(km.inertia_)

plt.plot(range(1, 11), elbow)
plt.xlabel("No. of Clusters")
plt.ylabel("Cost")
plt.show()

km = KM(n_clusters = 5)
res = km.fit_predict(X)

colors = ["red", "blue", "green", "yellow", "silver"]

for  i in range(5):
	plt.scatter(X[res == i,0], X[res == i,1], c=colors[i])
plt.axes().get_xaxis().set_visible(False)
plt.axes().get_yaxis().set_visible(False)
plt.show()