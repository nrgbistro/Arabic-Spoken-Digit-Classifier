import time
from matplotlib import pyplot as plt
from GMM import GaussianMixtureModel
from GradientColorMapper import GradientColorMapper
from Likelihood import Likelihood


class Classifier:
    def __init__(self, training_data, params):
        self.training_data = training_data
        self.params = params
        self.gmms = self._generate_models()
        self.Likelihood = Likelihood(self.gmms, self.params["cluster_nums"])

    def _generate_models(self):
        GMM = GaussianMixtureModel(self.training_data, self.params["cluster_nums"], self.params["use_kmeans"])
        return GMM.get_gmms()

    def _classify(self, utterance):
        return self.Likelihood.get_max_likelihood(utterance)

    def confusion(self, testing_data):
        print("Generating confusion matrix...")
        confusion_matrix = [[0 for _ in range(10)] for _ in range(10)]
        start_time = time.time()
        for digit in range(10):
            data = testing_data.filter_by_digit(digit)
            for utterance in data:
                confusion_matrix[digit][self._classify(utterance)] += 1
        end_time = time.time()
        print("Generated confusion matrix after " + "{:.2f}".format(round(end_time - start_time, 2)) + " seconds")
        MAX_CORRECT = 220
        diag_cmapper = GradientColorMapper((1, 0, 0), (0, 1, 0), MAX_CORRECT)
        offdiag_cmapper = GradientColorMapper((.6, .6, 1), (1, 1, 0), MAX_CORRECT)

        # Create a 2D list for cell colors
        cell_colors = [[offdiag_cmapper(confusion_matrix[i][j]) if i != j else diag_cmapper(confusion_matrix[i][j])
                        for j in range(len(confusion_matrix[i]))] for i in range(len(confusion_matrix))]

        fig, ax = plt.subplots()
        ax.axis('off')
        ax.table(cellText=confusion_matrix, cellColours=cell_colors, loc='center', cellLoc='center', colLabels=[f"GMM {i}" for i in range(10)],
                 rowLabels=[f"Test {i}" for i in range(10)])
        ax.set_title(f"Confusion Matrix")
        plt.show()
