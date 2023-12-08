import time
import numpy as np
from sklearn.cluster import KMeans
from ModifiedGaussianMixture import GaussianMixture
from Covariance import Covariance
from parsing.ParsedData import get_all_blocks

RANDOM_STATE = 42


def _convert_cov_constraints_em(covariance_type, covariance_tied):
    if covariance_type == "full":
        if covariance_tied:
            return "tied_full"
        else:
            return "full"
    elif covariance_type == "diagonal" or covariance_type == "diag":
        if covariance_tied:
            return "tied_diag"
        else:
            return "diag"
    elif covariance_type == "spherical":
        if covariance_tied:
            return "tied_spherical"
        else:
            return "spherical"
    else:
        raise ValueError("Invalid covariance type: " + covariance_type)


class GaussianMixtureModel:
    def __init__(self, data, hyperparams):
        self.data = data
        self.hyperparams = hyperparams

    def get_gmms(self):
        print("Training GMMs...")
        start_time = time.time()
        ret = []
        for i in range(10):
            ret.append(self.train_gmm(i))
        end_time = time.time()
        print("Training complete after " + "{:.2f}".format(round(end_time - start_time, 2)) + " seconds")
        return ret

    def train_gmm(self, digit):
        start_time = time.time()
        coordinates = np.array(get_all_blocks(self.data.filter_by_digit(digit)))
        labels, centers, cov = self._k_means(digit) if self.hyperparams["use_kmeans"] else self._em(digit)
        clusters = {}
        for i in range(self.hyperparams["k_mapping"][digit]):
            clusters[i] = []

        for i, label in enumerate(labels):
            clusters[label].append(coordinates[i])

        cluster_info = []
        for i in range(self.hyperparams["k_mapping"][digit]):
            pi = len(clusters[i]) / sum(len(cluster) for cluster in clusters.values())
            mean = centers[i]
            covariance = cov[i]
            cluster_info.append({
                "pi": pi,
                "mean": mean,
                "covariance": covariance
            })
        end_time = time.time()
        print("Trained digit " + str(digit) + " with k=" + str(
            self.hyperparams["k_mapping"][digit]) + " after " + "{:.2f}".format(
            round(end_time - start_time, 2)) + " seconds")
        return cluster_info

    def _k_means(self, digit):
        all_mfccs = np.vstack([data_block.mfccs for data_block in self.data.filter_by_digit(digit)])
        kmeans = KMeans(n_clusters=self.hyperparams["k_mapping"][digit], random_state=RANDOM_STATE, n_init='auto')
        kmeans.fit(all_mfccs)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        Cov = Covariance(self.hyperparams["covariance_type"], self.hyperparams["covariance_tied"])
        covariance = Cov(all_mfccs, labels, centers, self.hyperparams["k_mapping"][digit])
        return labels, centers, covariance

    def _em(self, digit):
        all_mfccs = np.vstack([data_block.mfccs for data_block in self.data.filter_by_digit(digit)])
        cov_constraint = _convert_cov_constraints_em(self.hyperparams["covariance_type"],
                                                     self.hyperparams["covariance_tied"])
        em = GaussianMixture(n_components=self.hyperparams["k_mapping"][digit], random_state=RANDOM_STATE, covariance_type=cov_constraint)
        em.fit(all_mfccs)
        labels = em.predict(all_mfccs)
        centers = em.means_
        covariance = self._fix_em_cov_output(em.covariances_, self.hyperparams["covariance_type"],
                                        self.hyperparams["covariance_tied"], self.hyperparams["k_mapping"][digit])

        return labels, centers, covariance

    def _fix_em_cov_output(self, covariance, covariance_type, covariance_tied, k):
        if covariance_type == "full":
            if covariance_tied:
                ret = np.asarray([covariance] * k)
            else:
                ret = covariance
        elif covariance_type == "diagonal" or covariance_type == "diag":
            if covariance_tied:
                ret = np.asarray([np.diag(cov) for cov in covariance])
            else:
                ret = np.asarray([np.diag(cov) for cov in covariance])
        elif covariance_type == "spherical":
            if covariance_tied:
                ret = np.asarray([np.diag([covariance] * 13)] * k)
            else:
                ret = np.asarray([np.diag([cov] * 13) for cov in covariance])
        else:
            ret = ValueError("Invalid covariance type: " + covariance_type)
        assert len(ret.shape) == 3
        assert ret.shape[0] == k
        assert ret.shape[1] == len(self.hyperparams["mfcc_indexes"])
        assert ret.shape[2] == len(self.hyperparams["mfcc_indexes"])
        return ret