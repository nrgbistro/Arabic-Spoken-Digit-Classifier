import time

import numpy as np
from matplotlib import pyplot as plt
from Classifier import Classifier
from parsing.ParsedData import ParsedData
from parsing.dataParser import parse_file


def test_all_covariance_types(hyperparams):
	training_data = ParsedData(parse_file("spoken_arabic_digits/Train_Arabic_Digit.txt", 66))
	testing_data = ParsedData(parse_file("spoken_arabic_digits/Test_Arabic_Digit.txt", 22))

	cov_types = ["spherical", "diagonal", "full"]

	output_mapping = {}

	for cov_type in cov_types:
		for cov_tied in [True, False]:
			print("Testing covariance type: " + cov_type + ", covariance tied: " + str(cov_tied))
			hyperparams["covariance_type"] = cov_type
			hyperparams["covariance_tied"] = cov_tied
			classifier = Classifier(training_data, hyperparams)
			avg_accuracy = classifier.confusion(testing_data, show_plot=False, show_timing=False)
			key = " ".join(["tied" if cov_tied else "distinct", cov_type])
			output_mapping[key] = avg_accuracy

	bar_names = [name.title() for name in list(output_mapping.keys())]
	bar_heights = list(output_mapping.values())

	fig, ax = plt.subplots()
	ax.bar(bar_names, bar_heights)
	ax.set_ylim(70, 100)
	ax.yaxis.grid(True)

	plt.yticks(np.arange(70, 100, 2))

	plt.xlabel('Covariance Constraint Type')
	plt.ylabel('Average Accuracy (%)')
	plot_title = "Average Accuracy Using " + ("K-Means" if hyperparams["use_kmeans"] else "Expectation Maximization")
	plt.title(plot_title)

	plt.show()


if __name__ == '__main__':
	hyperparameters = {
		"mfcc_indexes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
		"use_kmeans": False,
		"covariance_type": "full",
		"covariance_tied":  True,
		"k_mapping": {
			0: 4,
			1: 3,
			2: 3,
			3: 4,
			4: 3,
			5: 4,
			6: 4,
			7: 4,
			8: 4,
			9: 5
		}
	}

	test_all_covariance_types(hyperparameters)
