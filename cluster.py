import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def k_means(data, k):
    all_mfccs = np.vstack([data_block.mfccs for data_block in data])
    kmeans = KMeans(n_clusters=k, random_state=k, n_init='auto').fit(all_mfccs)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    covariance = kmeans.covariance_
    return labels, centers, covariance


def em(data, k):
    all_mfccs = np.vstack([data_block.mfccs for data_block in data])
    gmm = GaussianMixture(n_components=k, random_state=k).fit(all_mfccs)
    labels = gmm.predict(all_mfccs)
    centers = gmm.means_
    covariance = gmm.covariances_
    return labels, centers, covariance
