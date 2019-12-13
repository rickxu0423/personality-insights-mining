from sklearn.cluster import KMeans

def kmeans(X, n):

    # Initializing KMeans
    kmeans = KMeans(n_clusters=n)
    # Fitting with inputs
    kmeans = kmeans.fit(X)
    # Predicting the clusters
    labels = kmeans.predict(X)
    # Getting the cluster centers
    C = kmeans.cluster_centers_
    return labels, C
