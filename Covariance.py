import numpy as np


def _tied_spherical_covariance(coords, labels, centers, k):
	dim = coords.shape[1]

	all_cluster_points = np.concatenate([coords[labels == i] - centers[i] for i in range(k)], axis=0)
	tied_covariance = np.identity(dim) * np.var(all_cluster_points)

	return [tied_covariance] * k


def _tied_diagonal_covariance(coords, labels, centers, k):
	cluster_points = np.concatenate([coords[labels == i] - centers[i] for i in range(k)])
	shared_covariance = np.diag(np.var(cluster_points, axis=0))

	covariances = []
	for i in range(k):
		cluster_covariance = np.identity(coords.shape[1]) * shared_covariance
		covariances.append(cluster_covariance)
	return covariances


def _distinct_diagonal_covariance(coords, labels, centers, k):
	covariances = []
	for i in range(k):
		cluster_points = coords[labels == i] - centers[i]
		covariance = np.diag(np.var(cluster_points, axis=0))
		covariances.append(covariance)
	return covariances


def _tied_full_covariance(coords, labels, centers, k):
	cluster_points = np.concatenate([coords[labels == i] - centers[i] for i in range(k)])
	shared_covariance = np.cov(cluster_points, rowvar=False)

	covariances = [shared_covariance] * k
	return covariances


def _distinct_spherical_covariance(coords, labels, centers, k):
	covariances = []
	for i in range(k):
		cluster_points = coords[labels == i] - centers[i]
		spherical_covariance = np.identity(coords.shape[0]) * np.var(cluster_points)
		covariances.append(spherical_covariance)
	return covariances


def _distinct_full_covariance(coords, labels, centers, k):
	covariances = []
	for i in range(k):
		cluster_points = coords[labels == i] - centers[i]
		full_covariance = np.cov(cluster_points, rowvar=False)
		covariances.append(full_covariance)
	return covariances


class Covariance:
	def __init__(self, cov_type, tied):
		self.type = cov_type
		self.tied = tied

	def __call__(self, coords, labels, centers, k):
		if self.type == "spherical":
			if self.tied:
				return _tied_spherical_covariance(coords, labels, centers, k)
			else:
				return _distinct_spherical_covariance(coords, labels, centers, k)
		elif self.type == "diagonal" or self.type == "diag":
			if self.tied:
				return _tied_diagonal_covariance(coords, labels, centers, k)
			else:
				return _distinct_diagonal_covariance(coords, labels, centers, k)
		elif self.type == "full":
			if self.tied:
				return _tied_full_covariance(coords, labels, centers, k)
			else:
				return _distinct_full_covariance(coords, labels, centers, k)
		else:
			raise ValueError("Invalid covariance type: " + self.type)
