import numpy as np
from matplotlib import pyplot as plt
from Classifier import Classifier
from parsing.ParsedData import ParsedData
from parsing.dataParser import parse_file


def test_all_combinations(hyperparams):
	cov_types = ["spherical", "diagonal", "full"]

	output_mapping = {}

	for cov_type in cov_types:
		for cov_tied in [True, False]:
			key = " ".join(["tied" if cov_tied else "distinct", cov_type])
			output_mapping[key] = [0, 0]

	for use_kmeans in [True, False]:
		for cov_type in cov_types:
			for cov_tied in [True, False]:
				print("\nTesting covariance type: " + cov_type + ", covariance tied: " + str(cov_tied))
				hyperparams["use_kmeans"] = use_kmeans
				hyperparams["covariance_type"] = cov_type
				hyperparams["covariance_tied"] = cov_tied
				classifier = Classifier(training_data, hyperparams)
				avg_accuracy = classifier.confusion(testing_data, show_plot=False, show_timing=False)
				key = " ".join(["tied" if cov_tied else "distinct", cov_type])
				if use_kmeans:
					output_mapping[key][0] = avg_accuracy
				else:
					output_mapping[key][1] = avg_accuracy

	bar_names = [name.title() for name in list(output_mapping.keys())]
	# bar_values = np.array([[76.36363636 76.81818182]
						#  [86.09090909 76.40909091]
						#  [82.13636364 84.        ]
						#  [86.09090909 86.45454545]
						#  [88.09090909 88.72727273]
						#  [87.54545455 88.90909091]])
	bar_values = np.array([list(value) for value in output_mapping.values()])
	print(bar_values)
	bar_width = 0.35
	positions = np.arange(len(bar_names))

	fig, ax = plt.subplots()
	ax.bar(positions - bar_width / 2, bar_values[:, 0], bar_width, label='K-Means', color='blue')
	ax.bar(positions + bar_width / 2, bar_values[:, 1], bar_width, label='Expectation Maximization', color='orange')

	ax.set_xticks(positions)
	ax.set_xticklabels(bar_names)

	ax.legend(loc='upper left')
	ax.set_ylim(75, 100)
	ax.yaxis.grid(True)

	plt.yticks(np.arange(75, 100, 1))
	plt.xlabel('Covariance Constraint Type')
	plt.ylabel('Average Accuracy (%)')
	plot_title = "Average Accuracy of All Hyperparameter Combinations"
	plt.title(plot_title)

	plt.show()


if __name__ == '__main__':
	training_data = ParsedData(parse_file("spoken_arabic_digits/Train_Arabic_Digit.txt", 66))
	testing_data = ParsedData(parse_file("spoken_arabic_digits/Test_Arabic_Digit.txt", 22))

	hyperparameters = {
		"mfcc_indexes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
		"use_kmeans": True,
		"covariance_type": "spherical",
		"covariance_tied":  False,
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

	# avg_accuracy = Classifier(training_data, hyperparameters).confusion(testing_data, show_plot=True, show_timing=True)
	# plt.show()

	test_all_combinations(hyperparameters)
