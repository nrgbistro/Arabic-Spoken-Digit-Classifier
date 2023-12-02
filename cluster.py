import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class Cluster:
	def __init__(self, data, k, algorithm):
		self.data = data
		self.k = k
		self.algorithm = algorithm

	def run(self):
		if self.algorithm == "kmeans":
			return self.k_means()
		elif self.algorithm == "em":
			return self.em()
		else:
			raise Exception("Invalid clustering algorithm")

	def k_means(self):
		all_mfccs = np.vstack([data_block.mfccs for data_block in self.data])
		kmeans = KMeans(n_clusters=self.k, random_state=self.k, n_init='auto').fit(all_mfccs)
		labels = kmeans.labels_
		centers = kmeans.cluster_centers_
		covariance = kmeans.covariance_
		return labels, centers, covariance

	def em(self):
		all_mfccs = np.vstack([data_block.mfccs for data_block in self.data])
		gmm = GaussianMixture(n_components=self.k, random_state=self.k).fit(all_mfccs)
		labels = gmm.predict(all_mfccs)
		centers = gmm.means_
		covariance = gmm.covariances_
		return labels, centers, covariance


def generate_gmm(data, k, clustering_algorithm):
	cluster = Cluster(data, k, clustering_algorithm)
	labels, centers, covariance = cluster.run()
	return {
		"labels": labels,
		"centers": centers,
		"covariance": covariance
	}
