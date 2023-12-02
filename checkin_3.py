import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from parsing.ParsedData import get_all_blocks

K = 4
rand = np.random.RandomState(42)


def k_means(coords):
    kmeans = KMeans(n_clusters=K, random_state=rand, n_init="auto", init="k-means++")
    kmeans.fit(coords)
    return kmeans.cluster_centers_, kmeans.labels_


def make_scatter_plot(coordinates, labels=None):
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, alpha=0.4)


def tied_spherical_covariance(coords, labels, centers):
    dim = coords.shape[1]

    all_cluster_points = np.concatenate([coords[labels == i] - centers[i] for i in range(K)], axis=0)
    tied_covariance = np.identity(dim) * np.var(all_cluster_points)

    return [tied_covariance] * K


def tied_diagonal_covariance(coords, labels, centers):
    cluster_points = np.concatenate([coords[labels == i] - centers[i] for i in range(K)])
    shared_covariance = np.diag(np.var(cluster_points, axis=0))

    covariances = []
    for i in range(K):
        cluster_covariance = np.identity(coords.shape[1]) * shared_covariance
        covariances.append(cluster_covariance)
    return covariances


def distinct_diagonal_covariance(coords, labels, centers):
    covariances = []
    for i in range(K):
        cluster_points = coords[labels == i] - centers[i]
        covariance = np.diag(np.var(cluster_points, axis=0))
        covariances.append(covariance)
    return covariances


def tied_full_covariance(coords, labels, centers):
    cluster_points = np.concatenate([coords[labels == i] - centers[i] for i in range(K)])
    shared_covariance = np.cov(cluster_points, rowvar=False)

    covariances = [shared_covariance] * K
    return covariances


def run_gmm(coordinates, digit):
    coordinates = np.array(get_all_blocks(coordinates.filter_by_digit(digit)))
    centers, labels = k_means(coordinates)
    clusters = {}
    for i in range(K):
        clusters[i] = []

    for i, label in enumerate(labels):
        clusters[label].append(coordinates[i])

    cov = tied_spherical_covariance(coordinates, labels, centers)

    cluster_info = []
    for i in range(K):
        pi = len(clusters[i]) / sum(len(cluster) for cluster in clusters.values())
        mean = centers[i]
        covariance = cov[i]
        cluster_info.append({
            "pi": pi,
            "mean": mean,
            "covariance": covariance
        })
    return cluster_info


def part_b_wrapper(data):
    target_cluster_info = run_gmm(data, 6)
    likelihoods = []
    for digit in range(10):
        likelihoods.append(likelihood(data, digit, target_cluster_info))

    fig, axs = plt.subplots(2, 5, tight_layout=True, sharex=True, sharey=True)

    for i, ax in enumerate(axs.flat):
        ax.set_title(f"Digit {i}")
        kd = KernelDensity(kernel='gaussian', bandwidth=40)
        likely = np.array(likelihoods[i]).reshape(-1, 1)
        kd.fit(np.array(likely))
        x = np.linspace(min(likely), max(likely), 1000).reshape(-1, 1)
        log_dens = kd.score_samples(x)
        ax.plot(x, np.exp(log_dens))

    plt.show()


def likelihood(data, digit, target_cluster_info):
    filtered_data = data.filter_by_digit(digit)
    ret = []
    for utterance in range(len(filtered_data)):
        sums = []
        for i in range(K):
            pi = target_cluster_info[i]["pi"]
            mean = target_cluster_info[i]["mean"]
            covariance = target_cluster_info[i]["covariance"]
            sums.append(pi * multivariate_normal.pdf(filtered_data.get()[utterance].mfccs, mean=mean, cov=covariance))
        out_sums = np.sum(np.array(sums), axis=0)
        ret.append(np.sum(np.log(out_sums)))
    return ret





