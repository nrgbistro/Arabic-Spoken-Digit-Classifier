import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class GaussianMixtureModel:
	def __init__(self, algorithm):
		self.algorithm = algorithm
		self.data = None
		self.k = None
		self.trained_gmm = None

	def train(self, k, data):
		self.data = data
		self.k = k
		if self.algorithm == "kmeans":
			self.k_means()
		elif self.algorithm == "em":
			self.em()
		else:
			raise Exception("Invalid clustering algorithm")

	def predict(self, new_point):
		return self.trained_gmm.predict(new_point)

	def k_means(self):
		all_mfccs = np.vstack([data_block.mfccs for data_block in self.data])
		kmeans = KMeans(n_clusters=self.k, random_state=self.k, n_init='auto').fit(all_mfccs)
		labels = kmeans.labels_
		centers = kmeans.cluster_centers_
		covariance = kmeans.covariance_
		self.trained_gmm = kmeans
		return labels, centers, covariance

	def em(self):
		all_mfccs = np.vstack([data_block.mfccs for data_block in self.data])
		gmm = GaussianMixture(n_components=self.k, random_state=self.k).fit(all_mfccs)
		labels = gmm.predict(all_mfccs)
		centers = gmm.means_
		covariance = gmm.covariances_
		self.trained_gmm = gmm
		return labels, centers, covariance


def generate_gmm(data, k, clustering_algorithm):
	gmm = GaussianMixtureModel(clustering_algorithm)
	gmm.train(data, k)
	return gmm
