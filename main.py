from Classifier import Classifier
from parsing.ParsedData import ParsedData
from parsing.dataParser import parse_file


if __name__ == '__main__':
	training_data = ParsedData(parse_file("spoken_arabic_digits/Train_Arabic_Digit.txt", 66))
	testing_data = ParsedData(parse_file("spoken_arabic_digits/Test_Arabic_Digit.txt", 22))

	hyperparameters = {
		"mfcc_indexes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
		"use_kmeans": False,
		"covariance_type": "diag",
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
			9: 4
		}
	}
	classifier = Classifier(training_data, hyperparameters)
	classifier.confusion(testing_data)
