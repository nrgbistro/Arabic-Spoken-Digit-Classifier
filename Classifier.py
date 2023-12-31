import time
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from GMM import GaussianMixtureModel


class Classifier:
    def __init__(self, training_data, hyperparams):
        self.training_data = training_data.filter_by_mfccs(hyperparams["mfcc_indexes"])
        self.hyperparams = hyperparams
        self.gmms = self.generate_models()

    def _compute_likelihoods(self, utterance):
        ret = []
        for i, gmm in enumerate(self.gmms):
            sums = []
            for k in range(self.hyperparams["k_mapping"][i]):
                pi = gmm[k]["pi"]
                mean = gmm[k]["mean"]
                covariance = gmm[k]["covariance"]

                # Precompute the log likelihood of the covariance determinant
                log_det_cov = -0.5 * np.log(np.linalg.det(covariance))

                # Compute the Mahalanobis distance without the determinant term
                mahalanobis_dist = np.sum(
                    (utterance.mfccs - mean) @ np.linalg.inv(covariance) * (utterance.mfccs - mean), axis=1)

                # Compute the log likelihood for the current component
                component_likelihood = pi * np.exp(log_det_cov - 0.5 * mahalanobis_dist)
                sums.append(component_likelihood)

            # Sum over all components for the current GMM
            out_sums = np.sum(np.array(sums), axis=0)

            # Compute the log likelihood for the current GMM
            ret.append(np.sum(np.log(out_sums)))

        return ret

    def _get_max_likelihood(self, utterance):
        return np.argmax(self._compute_likelihoods(utterance))

    def generate_models(self):
        GMM = GaussianMixtureModel(self.training_data, self.hyperparams)
        return GMM.get_gmms()

    def confusion_row(self, testing_data, digit):
        ret = [0] * 10
        data = testing_data.filter_by_digit(digit)
        for utterance in data:
            ret[self._get_max_likelihood(utterance)] += 1
        return ret

    def confusion(self, testing_data, show_plot=True, show_timing=False):
        MAX_CORRECT = round(len(testing_data.get()) / 10)
        testing_data = testing_data.filter_by_mfccs(self.hyperparams["mfcc_indexes"])
        if show_timing:
            print("Generating confusion matrix...")
        confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]
        for digit in range(10):
            start_time = time.time()
            confusion_matrix[digit] = self.confusion_row(testing_data, digit)
            end_time = time.time()
            if show_timing:
                print(f"Generated row {str(digit)} after " + "{:.2f}".format(round(end_time - start_time, 2)) + " seconds")
        if show_plot:
            self.plot_confusion(confusion_matrix, max_correct=MAX_CORRECT)
        accuracy_percentages = [confusion_matrix[i][i] / MAX_CORRECT * 100 for i in range(len(confusion_matrix))]
        pprint(confusion_matrix)
        return accuracy_percentages, np.mean(accuracy_percentages)

    def plot_confusion(self, confusion_matrix, max_correct=220):
        fig, (ax_matrix, ax_colorbar) = plt.subplots(1, 2,
                                                 gridspec_kw={'width_ratios': [4, .1]},
                                                 figsize=(9, 8))
        confusion_matrix_percentage = [[confusion_matrix[i][j] / max_correct * 100 for j in range(len(confusion_matrix[i]))] for i in range(len(confusion_matrix))]
        cmap = LinearSegmentedColormap.from_list('custom_colormap', [(.95, 0, 0), (0, .8, 0)], N=256)
        im = ax_matrix.imshow(confusion_matrix_percentage, cmap=cmap, vmin=0, vmax=100)
        ax_matrix.set_xticks(np.arange(10))
        ax_matrix.set_yticks(np.arange(10))
        ax_matrix.set_xticklabels([str(i) for i in range(10)])
        ax_matrix.set_yticklabels([str(i) for i in range(10)])
        ax_matrix.set_xlabel("Predicted Values")
        ax_matrix.set_ylabel("True Values")

        avg_accuracy = np.mean([confusion_matrix[i][i] / max_correct * 100 for i in range(len(confusion_matrix))])

        for i in range(10):
            for j in range(10):
                ax_matrix.text(j, i, "{:.2f}%".format(confusion_matrix_percentage[i][j]), ha='center', va='center', color='white')

        ax_matrix.text(4.5, 10.5, f'Average Accuracy: {avg_accuracy:.2f}%', ha='center', va='center', color='black')

        colorbar_ticks = np.arange(0, 101, 10)
        cbar = fig.colorbar(im, cax=ax_colorbar)
        cbar.set_ticks(colorbar_ticks)
        cbar.set_ticklabels(colorbar_ticks)
        if self.hyperparams["covariance_type"] == "diagonal" or self.hyperparams["covariance_type"] == "diag":
            covariance_type_string = "Diagonal"
        elif self.hyperparams["covariance_type"] == "full":
            covariance_type_string = "Full"
        else:
            covariance_type_string = "Spherical"
        covariance_tied_string = "Tied" if self.hyperparams["covariance_tied"] else "Distinct"
        algorithm_string = "K-Means" if self.hyperparams["use_kmeans"] else "EM"
        fig.suptitle(" ".join(
            ["Confusion Matrix Using", algorithm_string, "With", covariance_tied_string, covariance_type_string,
             "Covariance"]))
