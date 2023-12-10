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
    coordinates = np.array(get_all_blocks(data.filter_by_digit(digit)))
    (labels, centers, cov) = GaussianMixtureModel(data.filter_by_digit(digit), k).train_gmm(digit)
    clusters = {}
    for i in range(k):
        clusters[i] = []

    for i, label in enumerate(labels):
        clusters[label].append(coordinates[i])

    cluster_info = []
    for i in range(k):
        pi = len(clusters[i]) / sum(len(cluster) for cluster in clusters.values())
        mean = centers[i]
        covariance = cov[i]
        cluster_info.append({
            "pi": pi,
            "mean": mean,
            "covariance": covariance
        })
    return cluster_info


def part_b_wrapper(data, params, d):
    k = params["cluster_nums"][d]
    target_cluster_info = run_gmm(data, d, k)
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
        ax.plot(x, y)
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





