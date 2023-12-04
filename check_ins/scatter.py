import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

from GMM import GaussianMixtureModel


def create_contours(mean, cov, points):
    x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
    y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1
    x_mesh, y_mesh = np.meshgrid(
        np.linspace(x_min, x_max, 1000),
        np.linspace(y_min, y_max, 1000))
    mvn = multivariate_normal(mean=mean, cov=cov)
    pdf_values = mvn.pdf(np.dstack((x_mesh, y_mesh)))
    return x_mesh, y_mesh, pdf_values


def create_scatter(data, k, digit):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    size = 0.75
    point_alpha = 0.5
    contour_alpha = 0.8

    point_relations = [[1, 0], [2, 0], [2, 1]]
    data_blocks = data.filter_by_digit(digit)
    all_mfccs = np.vstack([data_block.mfccs for data_block in data_blocks])
    gmm = GaussianMixtureModel(data_blocks, k, use_kmeans=False)
    labels, centers, covariance = gmm.em()

    v0 = all_mfccs[:, 0]
    v1 = all_mfccs[:, 1]
    v2 = all_mfccs[:, 2]

    points = [v0, v1, v2]
    for i, ax in enumerate(axes.flat):
        clusters = [all_mfccs[labels == i] for i in range(k)]

        contours = []
        ax.scatter(points[point_relations[i][0]], points[point_relations[i][1]], c=labels, s=size, alpha=point_alpha)
        ax.set_title(f"MFCC {point_relations[i][1] + 1} (y) vs MFCC {point_relations[i][0] + 1} (x)")
        for j, cluster in enumerate(clusters):
            cov = np.cov(cluster[:, point_relations[i]], rowvar=False)
            contours.append(create_contours(centers[j, point_relations[i]], cov, cluster[:, point_relations[i]]))
        for j in range(k):
            ax.contour(contours[j][0], contours[j][1], contours[j][2], colors='black', alpha=contour_alpha)

    fig.suptitle(f"Digit {digit}")
    plt.tight_layout()
    plt.show()
