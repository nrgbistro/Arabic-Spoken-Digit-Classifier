from cluster import k_means, em, generate_gmm


class Classifier:
	def __init__(self, training_data, testing_data, params):
		self.training_data = training_data
		self.testing_data = testing_data
		self.params = params

	def generate_model(self):
		gmms = []
		clustering_algorithm = k_means if self.params["use_kmeans"] else em
		for i in range(10):
			k = self.params["cluster_nums"][i]
			gmms.append(generate_gmm(self.training_data.filter_by_digit(i), k, clustering_algorithm))
		return gmms

	def run(self):
		pass
