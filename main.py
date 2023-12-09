import random
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from Classifier import Classifier
from check_ins.checkin_1 import make_plots_part_a
from parsing.ParsedData import ParsedData
from parsing.dataParser import parse_file

cov_types = ["spherical", "diagonal", "full"]


def test_all_combinations_avg(hyperparams):
    output_mapping = {}
    for cov_type in cov_types:
        for cov_tied in [True, False]:
            key = " ".join(["tied" if cov_tied else "distinct", cov_type])
            output_mapping[key] = [0, 0]

    # for use_kmeans in [True, False]:
    #     for cov_type in cov_types:
    #         for cov_tied in [True, False]:
    #             print("\nTesting covariance type: " + cov_type + ", covariance tied: " + str(cov_tied))
    #             hyperparams["use_kmeans"] = use_kmeans
    #             hyperparams["covariance_type"] = cov_type
    #             hyperparams["covariance_tied"] = cov_tied
    #             classifier = Classifier(training_data, hyperparams)
    #             _, avg_accuracy = classifier.confusion(testing_data, show_plot=False, show_timing=False)
    #             key = " ".join(["tied" if cov_tied else "distinct", cov_type])
    #             if use_kmeans:
    #                 output_mapping[key][0] = avg_accuracy
    #             else:
    #                 output_mapping[key][1] = avg_accuracy

    bar_names = [name.title() for name in list(output_mapping.keys())]
    bar_values = np.array([list(value) for value in output_mapping.values()])
    bar_values = np.array([[74.72727273, 74.86363636],
                           [84.54545455, 73.72727273],
                           [80.54545455, 81.95454545],
                           [84.54545455, 84.68181818],
                           [86.5       , 88.09090909],
                           [88.        , 90.        ]])
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


def test_mfcc_combinations(hyperparams, training, testing):
    current_mfcc_indexes = []
    accuracies = []
    for num_mfccs in range(13):
        best_new_mfcc_index = -1
        best_new_mfcc_accuracy = 0
        new_mfcc_indexes = current_mfcc_indexes.copy()
        for new_mfcc_index in range(13):
            if new_mfcc_index in new_mfcc_indexes:
                continue
            else:
                new_mfcc_indexes.append(new_mfcc_index)
                print("testing mfcc indexes: " + str(new_mfcc_indexes) + "...")
                hyperparams["mfcc_indexes"] = new_mfcc_indexes
                classifier = Classifier(training, hyperparams)
                _, avg_accuracy = classifier.confusion(testing, show_plot=False, show_timing=False)
                # avg_accuracy = random.randint(1, 100)
                if avg_accuracy > best_new_mfcc_accuracy:
                    best_new_mfcc_accuracy = avg_accuracy
                    best_new_mfcc_index = new_mfcc_index

                new_mfcc_indexes.pop()
        current_mfcc_indexes.append(best_new_mfcc_index)
        new_mfcc_indexes.append(best_new_mfcc_index)
        accuracies.append([best_new_mfcc_accuracy, sorted(new_mfcc_indexes)])

    female_accuracies = [[58.45454545454545, [4]],
                         [81.72727272727272, [2, 4]],
                         [88.18181818181819, [1, 2, 4]],
                         [92.45454545454547, [1, 2, 4, 7]],
                         [93.36363636363636, [1, 2, 4, 7, 10]],
                         [94.0909090909091, [1, 2, 4, 6, 7, 10]],
                         [94.0909090909091, [1, 2, 3, 4, 6, 7, 10]],
                         [93.1818181818182, [1, 2, 3, 4, 5, 6, 7, 10]],
                         [93.9090909090909, [1, 2, 3, 4, 5, 6, 7, 10, 12]],
                         [93.18181818181819, [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]],
                         [92.81818181818183, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]],
                         [92.0909090909091, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]],
                         [90.18181818181817, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]
    male_accuracies = [[35.909090909090914, [7]],
                         [58.27272727272727, [7, 10]],
                         [70.63636363636364, [6, 7, 10]],
                         [80.54545454545456, [4, 6, 7, 10]],
                         [83.81818181818183, [4, 6, 7, 8, 10]],
                         [88.27272727272728, [3, 4, 6, 7, 8, 10]],
                         [89.0909090909091, [2, 3, 4, 6, 7, 8, 10]],
                         [89.9090909090909, [2, 3, 4, 6, 7, 8, 9, 10]],
                         [89.27272727272728, [0, 2, 3, 4, 6, 7, 8, 9, 10]],
                         [90.54545454545453, [0, 2, 3, 4, 6, 7, 8, 9, 10, 12]],
                         [87.72727272727272, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]],
                         [87.0909090909091, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
                         [84.81818181818181, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]
    # accuracies = male_accuracies
    pprint(accuracies)
    x = [i for i in range(13)]
    y = [accuracy[0] for accuracy in accuracies]
    max_index = y.index(max(y))
    plt.figure()
    plt.plot(x, y, marker='o', zorder=1, markersize=5)
    plt.scatter(max_index, y[max_index], color='red', marker='^', label='Max Point = ' + str(round(max(y), 2)) + "%", zorder=2, s=150)
    plt.xlabel("Number of MFCCs")
    plt.ylabel("Average Accuracy (%)")
    plt.title("Analysis of Number of MFCC Indices")
    plt.legend(loc="upper left", fontsize=14)
    plt.xticks(x)


def test_k_values(hyperparams):
    k_mapping = hyperparams["k_mapping"]
    accuracies = []
    for digit in range(10):
        rng = [k_mapping[digit] - 1, k_mapping[digit], k_mapping[digit] + 1]
        best_k = -1
        best_accuracy = 0
        for k in rng:
            hyperparams["k_mapping"][digit] = k
            classifier = Classifier(training_data, hyperparams)
            _, avg_accuracy = classifier.confusion(testing_data, show_plot=False, show_timing=False)
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_k = k
        accuracies.append([digit, best_accuracy, best_k])
    pprint(accuracies)


if __name__ == '__main__':
    training_data = ParsedData(parse_file("spoken_arabic_digits/Train_Arabic_Digit.txt", 66))
    testing_data = ParsedData(parse_file("spoken_arabic_digits/Test_Arabic_Digit.txt", 22))

    all_mfccs = [i for i in range(13)]

    hyperparameters = {
        "mfcc_indexes": [1, 2, 4, 5, 6, 7, 8, 10, 12],
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

    # confusion = [[211, 0, 1, 0, 0, 0, 5, 2, 0, 1],
    #              [0, 206, 4, 0, 0, 1, 0, 7, 0, 2],
    #              [1, 3, 169, 0, 0, 18, 9, 9, 8, 3],
    #              [5, 0, 2, 192, 0, 0, 0, 18, 3, 0],
    #              [1, 11, 0, 0, 191, 0, 0, 13, 0, 4],
    #              [0, 0, 0, 0, 2, 203, 2, 5, 2, 6],
    #              [9, 0, 0, 0, 0, 0, 205, 0, 0, 6],
    #              [4, 0, 0, 2, 3, 7, 0, 197, 0, 7],
    #              [0, 6, 3, 3, 0, 3, 1, 0, 201, 3],
    #              [0, 0, 0, 1, 0, 8, 6, 1, 0, 204]]

    # test_all_combinations_avg(hyperparameters)
    # test_all_combinations_individual(hyperparameters)
    # test_k_values(hyperparameters)
    # male_train = training_data.filter_by_gender("M")
    # male_test = testing_data.filter_by_gender("M")
    # female_train = training_data.filter_by_gender("F")
    # female_test = testing_data.filter_by_gender("F")
    # test_mfcc_combinations(hyperparameters, training_data, testing_data)
    # test_mfcc_combinations(hyperparameters, female_train, female_test)
    Classifier(training_data, hyperparameters).confusion(testing_data, show_plot=True, show_timing=True)
    # Classifier(training_data, hyperparameters).plot_confusion(confusion)
    # print(Classifier(female_train, hyperparameters).confusion(female_test, show_plot=True, show_timing=True))
    # make_plots_part_a(training_data)
    plt.show()

