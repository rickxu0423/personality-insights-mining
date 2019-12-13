import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

idList = ['Openness', 'Adventurousness', 'Artistic Interests', 'Experiencing Emotions', 'Creative Thinking', 'Need for Cognition', 'Questioning', 'Conscientiousness', 'Achievement Striving', 'Cautiousness', 'Dutifulness', 'Orderliness', 'Self-discipline', 'Self-efficacy', 'Extraversion', 'Active', 'Leadership', 'Cheerfulness', 'Need for Stimulation', 'Outgoing', 'Social', 'Agreeableness', 'Altruism', 'Cooperation', 'Modesty', 'Forthright', 'Compassion', 'Trust', 'Emotional Response', 'Easy to Provoke', 'Anxious', 'Despondence', 'Self-control', 'Self-monitoring', 'Stress Management']
labelList = ['Component 1', 'Component 2', 'Component 3']
file = "data/personality.csv"
plt.rcParams["figure.figsize"] = (16, 9)

# PCA process
df = pd.read_csv(file)
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

# k means determine k
distortions = []
K = range(1,30)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])


# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')

plt.savefig('elbow.png')