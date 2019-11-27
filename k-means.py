import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

idList = ['Openness', 'Adventurousness', 'Artistic Interests', 'Experiencing Emotions', 'Creative Thinking', 'Need for Cognition', 'Questioning', 'Conscientiousness', 'Achievement Striving', 'Cautiousness', 'Dutifulness', 'Orderliness', 'Self-discipline', 'Self-efficacy', 'Extraversion', 'Active', 'Leadership', 'Cheerfulness', 'Need for Stimulation', 'Outgoing', 'Social', 'Agreeableness', 'Altruism', 'Cooperation', 'Modesty', 'Forthright', 'Compassion', 'Trust', 'Emotional Response', 'Easy to Provoke', 'Anxious', 'Despondence', 'Self-control', 'Self-monitoring', 'Stress Management']

plt.rcParams["figure.figsize"] = (16, 9)

data = pd.read_csv("data/personality.csv")
print("Input Data and Shape")
print(data.shape)
data.head()

iterables = []
for i in range(len(idList)):
    iterables.append(data[idList[i]].values)
#f1 = iterables[0]
#f2 = iterables[1]
X = np.array(list(zip(*iterables)))


# Initializing KMeans
kmeans = KMeans(n_clusters=3)
# Fitting with inputs
kmeans = kmeans.fit(X)
# Predicting the clusters
labels = kmeans.predict(X)
# Getting the cluster centers
C = kmeans.cluster_centers_

fig = plt.figure()
ax = Axes3D(fig)
#ax.scatter(X[:, 0], X[:, 7], X[:, 14], c=labels)
#ax.scatter(C[:, 0], C[:, 7], C[:, 14], marker='*', c='#050505', s=1000)

#(0,Openness),(7,Conscientiousness),(14,Extraversion),(21,Agreeableness),(28,Emotional Response)

ax.scatter(X[:, 0], X[:, 7], X[:, 28], c=labels)
ax.scatter(C[:, 0], C[:, 7], C[:, 28], marker='*', c='#050505', s=1000)
plt.savefig('3.png')
print(C)