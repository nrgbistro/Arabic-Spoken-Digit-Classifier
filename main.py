import time
from matplotlib import pyplot as plt
from Classifier import Classifier
from parsing.ParsedData import ParsedData
from parsing.dataParser import parse_file


if __name__ == '__main__':
	training_data = ParsedData(parse_file("spoken_arabic_digits/Train_Arabic_Digit.txt", 66))
	testing_data = ParsedData(parse_file("spoken_arabic_digits/Test_Arabic_Digit.txt", 22))

	hyperparameters = {
		"mfcc_indexes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
		"use_kmeans": False,
		"covariance_type": "spherical",
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

	start_time = time.time()
	classifier = Classifier(training_data, hyperparameters)
	avg_accuracy = classifier.confusion(testing_data)
	print("Average accuracy: " + str(round(avg_accuracy, 2)) + "%")
	end_time = time.time()
	print("Total time: " + "{:.2f}".format(round(end_time - start_time, 2)) + " seconds")
	plt.show()

