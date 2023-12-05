import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from Covariance import Covariance
from parsing.ParsedData import get_all_blocks

RANDOM_STATE = 42


class GaussianMixtureModel:
	def __init__(self, data, hyperparams):
		self.data = data
		self.hyperparams = hyperparams

	def get_gmms(self):
		print("Training GMMs...")
		start_time = time.time()
		ret = []
		for i in range(10):
			ret.append(self.train_gmm(i))
		end_time = time.time()
		print("Training complete after " + "{:.2f}".format(round(end_time - start_time, 2)) + " seconds")
		return ret

	def train_gmm(self, digit):
		start_time = time.time()
		coordinates = np.array(get_all_blocks(self.data.filter_by_digit(digit)))
		labels, centers, cov = self._k_means(digit) if self.hyperparams["use_kmeans"] else self._em(digit)
		clusters = [[] for _ in range(self.hyperparams["k_mapping"][digit])]

		for i, label in enumerate(labels):
			clusters[label].append(coordinates[i])

		cluster_lengths = np.array([len(cluster) for cluster in clusters])
		pi = cluster_lengths / cluster_lengths.sum()
		cluster_info = [{"pi": pi[i], "mean": centers[i], "covariance": cov[i]} for i in range(len(pi))]

		end_time = time.time()
		print("Trained digit " + str(digit) + " with k=" + str(
			self.hyperparams["k_mapping"][digit]) + " after " + "{:.2f}".format(
			round(end_time - start_time, 2)) + " seconds")
		return cluster_info

	def _k_means(self, digit):
		all_mfccs = np.vstack([data_block.mfccs for data_block in self.data.filter_by_digit(digit)])
		kmeans = KMeans(n_clusters=self.hyperparams["k_mapping"][digit], random_state=RANDOM_STATE, n_init='auto')
		kmeans.fit(all_mfccs)
		labels = kmeans.labels_
		centers = kmeans.cluster_centers_
		Cov = Covariance(self.hyperparams["covariance_type"], self.hyperparams["covariance_tied"])
		covariance = Cov(all_mfccs, labels, centers, self.hyperparams["k_mapping"][digit])
		return labels, centers, covariance

	def _em(self, digit):
		all_mfccs = np.vstack([data_block.mfccs for data_block in self.data.filter_by_digit(digit)])
		em = GaussianMixture(n_components=self.hyperparams["k_mapping"][digit], random_state=RANDOM_STATE)
		em.fit(all_mfccs)
		labels = em.predict(all_mfccs)
		centers = em.means_
		covariance = em.covariances_
		return labels, centers, covariance
