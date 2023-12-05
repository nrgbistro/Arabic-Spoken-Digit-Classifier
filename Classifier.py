import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from GMM import GaussianMixtureModel
from GradientColorMapper import GradientColorMapper

MAX_CORRECT = 220


class Classifier:
    def __init__(self, training_data, hyperparams):
        self.training_data = training_data
        self.hyperparams = hyperparams
        self.gmms = self.generate_models()

    def _compute_likelihoods(self, utterance):
        ret = []
        for i in range(len(self.gmms)):
            sums = []
            gmm = self.gmms[i]
            for k in range(self.hyperparams["k_mapping"][i]):
                pi = gmm[k]["pi"]
                mean = gmm[k]["mean"]
                covariance = gmm[k]["covariance"]
                sums.append(
                    pi * multivariate_normal.pdf(utterance.mfccs, mean=mean, cov=covariance))
            out_sums = np.sum(np.array(sums), axis=0)
            ret.append(np.sum(np.log(out_sums)))
        return ret

    def _get_max_likelihood(self, utterance):
        return np.argmax(self._compute_likelihoods(utterance))

    def generate_models(self):
        GMM = GaussianMixtureModel(self.training_data, self.hyperparams)
        return GMM.get_gmms()

    def confusion(self, testing_data):
        print("Generating confusion matrix...")
        confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]
        for digit in range(10):
            data = testing_data.filter_by_digit(digit)
            start_time = time.time()

            for utterance in data:
                confusion_matrix[digit][self._get_max_likelihood(utterance)] += 1
            end_time = time.time()
            print(f"Generated row {str(digit)} after " + "{:.2f}".format(round(end_time - start_time, 2)) + " seconds")
        diag_color_mapper = GradientColorMapper((1, 0, 0), (0, 1, 0), MAX_CORRECT)
        off_diag_color_mapper = GradientColorMapper((1, 1, 1), (0, 0, 0), MAX_CORRECT)

        # Create a 2D list for cell colors
        cell_colors = [[off_diag_color_mapper(confusion_matrix[i][j]) if i != j else diag_color_mapper(confusion_matrix[i][j])
                        for j in range(len(confusion_matrix[i]))] for i in range(len(confusion_matrix))]

        confusion_matrix_str = [[str(round(confusion_matrix[i][j] / MAX_CORRECT * 100, 2)) + "%" for j in range(len(confusion_matrix[i]))] for i in range(len(confusion_matrix))]
        avg_accuracy = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix))) / MAX_CORRECT * 10
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.table(cellText=confusion_matrix_str, cellColours=cell_colors, loc='center', cellLoc='center', colLabels=[f"GMM {i}" for i in range(10)],
                 rowLabels=[f"Test {i}" for i in range(10)])
        if self.hyperparams["covariance_type"] == "diagonal" or self.hyperparams["covariance_type"] == "diag":
            covariance_type_string = "Diagonal"
        elif self.hyperparams["covariance_type"] == "full":
            covariance_type_string = "Full"
        else:
            covariance_type_string = "Spherical"
        covariance_tied_string = "Tied" if self.hyperparams["covariance_tied"] else "Distinct"
        algorithm_string = "K-Means" if self.hyperparams["use_kmeans"] else "EM"
        ax.set_title(" ".join(["Confusion Matrix Using", algorithm_string, "With", covariance_tied_string, covariance_type_string, "Covariance"]))
        return avg_accuracy
