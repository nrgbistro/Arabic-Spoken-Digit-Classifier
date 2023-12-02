from GMM import GaussianMixtureModel


class Classifier:
	def __init__(self, training_data, params):
		self.training_data = training_data
		self.params = params

	def generate_model(self):
		gmms = []
		use_kmeans = self.params["use_kmeans"]
		for i in range(10):
			k = self.params["cluster_nums"][i]
			new_gmm = GaussianMixtureModel(self.training_data.filter_by_digit(i), k, use_kmeans)
			gmms.append(new_gmm)
		return gmms

	def test(self, testing_data, gmms):
		pass
