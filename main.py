from checkin_1 import make_plots_part_b, make_plots_part_a
from checkin_3 import part_b_wrapper
from Classifier import Classifier
from parsing.ParsedData import ParsedData
from parsing.dataParser import parse_file
from scatter import create_scatter


if __name__ == '__main__':
	training_data = ParsedData(parse_file("spoken_arabic_digits/Train_Arabic_Digit.txt"))
	testing_data = ParsedData(parse_file("spoken_arabic_digits/Test_Arabic_Digit.txt"))

	# make_plots_part_a(training_data)
	hyperparameters = {
		"mfcc_indexes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
		"use_kmeans": False,
		"covariance_type": "full",
		"covariance_tied": False,
		"cluster_nums": {
			0: 4,
			1: 3,
			2: 3,
			3: 4,
			4: 3,
			5: 3,
			6: 4,
			7: 3,
			8: 4,
			9: 3
		}
	}
	classifier = Classifier(training_data, testing_data, hyperparameters)
	classifier.run()
