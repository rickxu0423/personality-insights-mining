import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

idList = ['Openness', 'Adventurousness', 'Artistic Interests', 'Experiencing Emotions', 'Creative Thinking', 'Need for Cognition', 'Questioning', 'Conscientiousness', 'Achievement Striving', 'Cautiousness', 'Dutifulness', 'Orderliness', 'Self-discipline', 'Self-efficacy', 'Extraversion', 'Active', 'Leadership', 'Cheerfulness', 'Need for Stimulation', 'Outgoing', 'Social', 'Agreeableness', 'Altruism', 'Cooperation', 'Modesty', 'Forthright', 'Compassion', 'Trust', 'Emotional Response', 'Easy to Provoke', 'Anxious', 'Despondence', 'Self-control', 'Self-monitoring', 'Stress Management']
labelList = ['Component 1', 'Component 2', 'Component 3']
file = "data/personality-pca.csv"
plt.rcParams["figure.figsize"] = (16, 9)

# PCA process
df = pd.read_csv(file, names=idList)
x = df.loc[:, idList].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = labelList)

# k-means
iterables = []
for label in labelList:
    iterables.append(principalDf[label].values)
X = np.array(list(zip(*iterables)))

# Initializing KMeans
kmeans = KMeans(n_clusters=5)
# Fitting with inputs
kmeans = kmeans.fit(X)
# Predicting the clusters
labels = kmeans.predict(X)
# Getting the cluster centers
C = kmeans.cluster_centers_


# Plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)

ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
ax.set_title('3D PCA', fontsize = 20)

plt.savefig('pca.png')