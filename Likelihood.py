import numpy as np
from scipy.stats import multivariate_normal


class Likelihood:
    def __init__(self, gmms, k_mapping):
        self.gmms = gmms
        self.k_mapping = k_mapping

    def _compute_likelihoods(self, utterance):
        ret = []
        for i in range(len(self.gmms)):
            sums = []
            gmm = self.gmms[i]
            for k in range(self.k_mapping[i]):
                pi = gmm[k]["pi"]
                mean = gmm[k]["mean"]
                covariance = gmm[k]["covariance"]
                sums.append(
                    pi * multivariate_normal.pdf(utterance.mfccs, mean=mean, cov=covariance))
            out_sums = np.sum(np.array(sums), axis=0)
            ret.append(np.sum(np.log(out_sums)))
        return ret

    def get_max_likelihood(self, utterance):
        return np.argmax(self._compute_likelihoods(utterance))
