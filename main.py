from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from Classifier import Classifier
from parsing.ParsedData import ParsedData
from parsing.dataParser import parse_file

cov_types = ["spherical", "diagonal", "full"]


def test_all_combinations_avg(hyperparams):
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
                _, avg_accuracy = classifier.confusion(testing_data, show_plot=False, show_timing=False)
                key = " ".join(["tied" if cov_tied else "distinct", cov_type])
                if use_kmeans:
                    output_mapping[key][0] = avg_accuracy
                else:
                    output_mapping[key][1] = avg_accuracy

    bar_names = [name.title() for name in list(output_mapping.keys())]
    # bar_values = np.array([[76.36363636, 76.81818182],
    # 					 [86.09090909, 76.40909091],
    # 					 [82.13636364, 84.        ],
    # 					 [86.09090909, 86.45454545],
    # 					 [88.09090909, 88.72727273],
    # 					 [87.54545455, 88.90909091]])
    bar_values = np.array([list(value) for value in output_mapping.values()])
    pprint(bar_values)
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


def test_all_combinations_individual(hyperparams):
    # output = []
    # for digit in range(10):
    # 	output_mapping = {}
    # 	for cov_type in cov_types:
    # 		for cov_tied in [True, False]:
    # 			key = " ".join(["tied" if cov_tied else "distinct", cov_type])
    # 			output_mapping[key] = [0, 0]
    #
    # 	for use_kmeans in [True, False]:
    # 		for cov_type in cov_types:
    # 			for cov_tied in [True, False]:
    # 				print("\nTesting covariance type: " + cov_type + ", covariance tied: " + str(cov_tied))
    # 				hyperparams["use_kmeans"] = use_kmeans
    # 				hyperparams["covariance_type"] = cov_type
    # 				hyperparams["covariance_tied"] = cov_tied
    # 				classifier = Classifier(training_data, hyperparams)
    # 				row = classifier.confusion_row(testing_data, digit)
    # 				key = " ".join(["tied" if cov_tied else "distinct", cov_type])
    # 				if use_kmeans:
    # 					output_mapping[key][0] = row[digit] / 220 * 100
    # 				else:
    # 					output_mapping[key][1] = row[digit] / 220 * 100
    # 	output.append(output_mapping)
    output = [{'distinct diagonal': [86.36363636363636, 92.72727272727272],
               'distinct full': [87.27272727272727, 94.54545454545455],
               'distinct spherical': [86.36363636363636, 72.27272727272728],
               'tied diagonal': [85.9090909090909, 87.72727272727273],
               'tied full': [89.0909090909091, 91.81818181818183],
               'tied spherical': [75.9090909090909, 75.0]},
              {'distinct diagonal': [90.45454545454545, 91.36363636363637],
               'distinct full': [95.0, 94.54545454545455],
               'distinct spherical': [90.45454545454545, 81.81818181818183],
               'tied diagonal': [91.36363636363637, 94.54545454545455],
               'tied full': [92.72727272727272, 89.54545454545455],
               'tied spherical': [83.63636363636363, 82.72727272727273]},
              {'distinct diagonal': [80.0, 77.27272727272727],
               'distinct full': [77.27272727272727, 80.9090909090909],
               'distinct spherical': [80.0, 73.63636363636363],
               'tied diagonal': [75.45454545454545, 76.36363636363637],
               'tied full': [83.18181818181817, 85.9090909090909],
               'tied spherical': [68.18181818181817, 72.72727272727273]},
              {'distinct diagonal': [90.0, 88.63636363636364],
               'distinct full': [81.81818181818183, 84.0909090909091],
               'distinct spherical': [90.0, 78.63636363636364],
               'tied diagonal': [85.9090909090909, 88.18181818181819],
               'tied full': [86.81818181818181, 90.45454545454545],
               'tied spherical': [79.0909090909091, 79.0909090909091]},
              {'distinct diagonal': [79.54545454545455, 67.27272727272727],
               'distinct full': [87.72727272727273, 86.36363636363636],
               'distinct spherical': [79.54545454545455, 62.727272727272734],
               'tied diagonal': [70.9090909090909, 69.54545454545455],
               'tied full': [86.36363636363636, 89.0909090909091],
               'tied spherical': [62.727272727272734, 63.63636363636363]},
              {'distinct diagonal': [78.63636363636364, 82.27272727272728],
               'distinct full': [86.36363636363636, 90.0],
               'distinct spherical': [78.63636363636364, 76.36363636363637],
               'tied diagonal': [73.18181818181819, 71.36363636363636],
               'tied full': [80.45454545454545, 82.72727272727273],
               'tied spherical': [75.45454545454545, 76.36363636363637]},
              {'distinct diagonal': [95.0, 96.81818181818181],
               'distinct full': [97.72727272727273, 95.45454545454545],
               'distinct spherical': [95.0, 78.63636363636364],
               'tied diagonal': [82.72727272727273, 88.18181818181819],
               'tied full': [91.36363636363637, 91.36363636363637],
               'tied spherical': [81.36363636363636, 81.36363636363636]},
              {'distinct diagonal': [78.18181818181819, 87.72727272727273],
               'distinct full': [85.0, 80.9090909090909],
               'distinct spherical': [78.18181818181819, 69.54545454545455],
               'tied diagonal': [75.9090909090909, 82.27272727272728],
               'tied full': [85.9090909090909, 83.18181818181817],
               'tied spherical': [68.18181818181817, 69.0909090909091]},
              {'distinct diagonal': [85.9090909090909, 82.72727272727273],
               'distinct full': [86.36363636363636, 87.72727272727273],
               'distinct spherical': [85.9090909090909, 81.81818181818183],
               'tied diagonal': [83.63636363636363, 88.63636363636364],
               'tied full': [89.0909090909091, 87.72727272727273],
               'tied spherical': [82.27272727272728, 81.36363636363636]},
              {'distinct diagonal': [96.81818181818181, 97.72727272727273],
               'distinct full': [90.9090909090909, 94.54545454545455],
               'distinct spherical': [96.81818181818181, 88.63636363636364],
               'tied diagonal': [96.36363636363636, 93.18181818181817],
               'tied full': [95.9090909090909, 95.45454545454545],
               'tied spherical': [86.81818181818181, 86.81818181818181]}]
    pprint(output)

    legend = [
        {"name": "K-Means", "color": "blue"},
        {"name": "Expectation Maximization", "color": "orange"}
    ]

    fig, axes = plt.subplots(5, 2)
    for i, ax in enumerate(axes.flatten()):
        bar_names = [name.title() for name in list(output[i].keys())]
        bar_values = np.array([list(value) for value in output[i].values()])
        bar_width = 0.35
        positions = np.arange(len(bar_names))
        ax.bar(positions - bar_width / 2, bar_values[:, 0], bar_width, label=legend[0]["name"], color=legend[0]["color"])
        ax.bar(positions + bar_width / 2, bar_values[:, 1], bar_width, label=legend[1]["name"], color=legend[1]["color"])
        ax.set_ylim(75, 100)
        ax.yaxis.grid(True)
        ax.set_yticks(np.arange(75, 101, 5))
        if i % 2 == 0:
            ax.set_ylabel('Accuracy (%)')
        ax.set_title(f"Digit {i}")
        ax.set_xticks(positions)
        ax.set_xticklabels(bar_names, fontsize=7, rotation=22)
        # all_values = [value for sublist in output[i].values() for value in sublist]
        # max_index = np.argmax(all_values)
    plt.subplots_adjust(wspace=0.25, hspace=1.25)
    legend_handles = [Patch(label=legend[0]["name"], color=legend[0]["color"]), Patch(label=legend[1]["name"], color=legend[1]["color"])]
    fig.legend(handles=legend_handles, loc='center', bbox_to_anchor=(0.1, 0.95))


if __name__ == '__main__':
    training_data = ParsedData(parse_file("spoken_arabic_digits/Train_Arabic_Digit.txt", 66))
    testing_data = ParsedData(parse_file("spoken_arabic_digits/Test_Arabic_Digit.txt", 22))

    hyperparameters = {
        "mfcc_indexes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "use_kmeans": False,
        "covariance_type": "full",
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

    print(Classifier(training_data, hyperparameters).confusion(testing_data, show_plot=True, show_timing=True))
    # test_all_combinations_avg(hyperparameters)
    # test_all_combinations_individual(hyperparameters)
    plt.show()

