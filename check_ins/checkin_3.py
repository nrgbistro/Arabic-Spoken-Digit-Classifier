import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from GMM import GaussianMixtureModel
from parsing.ParsedData import get_all_blocks

rand = np.random.RandomState(42)


def k_means(coords, k):
    kmeans = KMeans(n_clusters=k, n_init="auto", init="k-means++")
    kmeans.fit(coords)
    return kmeans.cluster_centers_, kmeans.labels_


def make_scatter_plot(coordinates, labels=None):
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, alpha=0.4)


def tied_spherical_covariance(coords, labels, centers, k):
    dim = coords.shape[1]

    all_cluster_points = np.concatenate([coords[labels == i] - centers[i] for i in range(k)], axis=0)
    tied_covariance = np.identity(dim) * np.var(all_cluster_points)

    return [tied_covariance] * k


def tied_diagonal_covariance(coords, labels, centers, k):
    cluster_points = np.concatenate([coords[labels == i] - centers[i] for i in range(k)])
    shared_covariance = np.diag(np.var(cluster_points, axis=0))

    covariances = []
    for i in range(k):
        cluster_covariance = np.identity(coords.shape[1]) * shared_covariance
        covariances.append(cluster_covariance)
    return covariances


def distinct_diagonal_covariance(coords, labels, centers, k):
    covariances = []
    for i in range(k):
        cluster_points = coords[labels == i] - centers[i]
        covariance = np.diag(np.var(cluster_points, axis=0))
        covariances.append(covariance)
    return covariances


def tied_full_covariance(coords, labels, centers, k):
    cluster_points = np.concatenate([coords[labels == i] - centers[i] for i in range(k)])
    shared_covariance = np.cov(cluster_points, rowvar=False)

    covariances = [shared_covariance] * k
    return covariances


def run_gmm(data, digit, k):
    return GaussianMixtureModel(data.filter_by_digit(digit), k).train_gmm(digit)


def part_b_wrapper(data, params, dig):
    k = params["cluster_nums"][dig]
    target_cluster_info = run_gmm(data, dig, params)
    likelihoods = []
    for digit in range(10):
        likelihoods.append(likelihood(data, digit, target_cluster_info, k))

    fig, axs = plt.subplots(10, 1, tight_layout=True, sharex=True, sharey=True)

    for i, ax in enumerate(axs.flat):
        ax.set_title(f"Digit {i}")
        kd = KernelDensity(kernel='gaussian', bandwidth=50)
        likely = np.array(likelihoods[i]).reshape(-1, 1)
        kd.fit(np.array(likely))
        x = np.linspace(min(likely), max(likely), 1000).reshape(-1, 1)
        log_dens = kd.score_samples(x)
        y = np.exp(log_dens)
        if i == 5:
            ax.set_ylabel("Probability Density")
        if i == 9:
            ax.set_xlabel("Log Likelihood")
        ax.plot(x, y)
    plt.suptitle(f"Log Likelihood of All Digits Against Digit {dig}'s Model")
    plt.show()


def likelihood(data, digit, target_cluster_info, k):
    filtered_data = data.filter_by_digit(digit)
    ret = []
    for utterance in range(len(filtered_data)):
        sums = []
        for i in range(k):
            pi = target_cluster_info[i]["pi"]
            mean = target_cluster_info[i]["mean"]
            covariance = target_cluster_info[i]["covariance"]
            sums.append(pi * multivariate_normal.pdf(filtered_data.get()[utterance].mfccs, mean=mean, cov=covariance))
        out_sums = np.sum(np.array(sums), axis=0)
        ret.append(np.sum(np.log(out_sums)))
    return ret


if __name__ == '__main__':
    from parsing.dataParser import parse_file
    from parsing.ParsedData import ParsedData

    d = ParsedData(parse_file("../spoken_arabic_digits/Train_Arabic_Digit.txt", 66))
    p = {
        "cluster_nums": {
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
        },
        "covariance_type": "full",
        "covariance_tied": False,
        "use_kmeans": False,
        "mfcc_indexes": [i for i in range(13)],
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
    part_b_wrapper(d, p, 1)
