import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from kmeans import kmeans
from pca import pca

plt.rcParams["figure.figsize"] = (16, 9)

def plotPCAKmean(X, labels, C):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels)
    ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)

    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title('3D PCA', fontsize = 20)

    plt.savefig('pca-1.png')

def exportLabeledData(X, labels):
    iterator = iter(labels)
    with open("data/personality-labeled-1.csv", "w") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",")
        filewriter.writerow(["Component1","Component2","Component3","Label"])
        for line in X:
            line = np.append(line, [next(iterator)])
            filewriter.writerow(line)

def exportTestData(X):
    with open("data/predict.csv", "w") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",")
        filewriter.writerow(["Component1","Component2","Component3"])
        for line in X:
            filewriter.writerow(line)


# pca
#file = "data/personality.csv"
file = "data/predict-nopca.csv"
X = pca(file)
# kmeans

#n = 5
#labels, C = kmeans(X, n)

# Plot
#plotPCAKmean(X, labels, C)
# Export labeled personality data
#exportLabeledData(X, labels)
exportTestData(X)

