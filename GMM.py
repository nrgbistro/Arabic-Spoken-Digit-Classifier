import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class GaussianMixtureModel:
	def __init__(self, data, k, use_kmeans=False):
		self.data = data
		self.k = k
		self.trained_gmm = None
		self.gmm_data = None
		if use_kmeans:
			self.gmm_data = self._k_means()
		else:
			self.gmm_data = self._em()

	def predict(self, new_point):
		return self.trained_gmm.predict(new_point)

	def get_gmm_data(self):
		return self.gmm_data

	def _k_means(self):
		all_mfccs = np.vstack([data_block.mfccs for data_block in self.data])
		kmeans = KMeans(n_clusters=self.k, random_state=self.k, n_init='auto').fit(all_mfccs)
		labels = kmeans.labels_
		centers = kmeans.cluster_centers_
		covariance = kmeans.covariance_
		self.trained_gmm = kmeans
		self.gmm_data = (labels, centers, covariance)
		return labels, centers, covariance

	def _em(self):
		all_mfccs = np.vstack([data_block.mfccs for data_block in self.data])
		em = GaussianMixture(n_components=self.k, random_state=self.k).fit(all_mfccs)
		labels = em.predict(all_mfccs)
		centers = em.means_
		covariance = em.covariances_
		self.trained_gmm = em
		self.gmm_data = (labels, centers, covariance)
		return labels, centers, covariance
