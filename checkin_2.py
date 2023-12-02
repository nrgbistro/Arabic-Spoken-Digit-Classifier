import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

point_alpha = 0.7
contour_alpha = 0.5
size = 0.4


def create_contours(mean, cov):
    x_mesh, y_mesh = np.meshgrid(
        np.linspace(mean[0] - 5, mean[0] + 5, 100),
        np.linspace(mean[1] - 5, mean[1] + 5, 100))
    mvn = multivariate_normal(mean=mean, cov=cov)
    pdf_values = mvn.pdf(np.dstack((x_mesh, y_mesh)))
    return x_mesh, y_mesh, pdf_values


def make_plots(data, digit=0, k=4):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    point_relations = [[1, 0], [2, 0], [2, 1]]
    for i, ax in enumerate(axes.flat):
        data_blocks = data.filter_by_digit(digit)
        all_mfccs = np.vstack([data_block.mfccs for data_block in data_blocks])

        kmeans = KMeans(n_clusters=k, random_state=digit*k, n_init='auto').fit(all_mfccs)
        labels = kmeans.labels_

        v0 = all_mfccs[:, 0]
        v1 = all_mfccs[:, 1]
        v2 = all_mfccs[:, 2]

        points = [v0, v1, v2]

        clusters = [all_mfccs[labels == i] for i in range(k)]

        contours = []
        ax.scatter(points[point_relations[i][0]], points[point_relations[i][1]], c=labels, s=size, alpha=point_alpha)
        ax.set_title(f"MFCC {point_relations[i][1] + 1} (y) vs MFCC {point_relations[i][0] + 1} (x)")
        for j, cluster in enumerate(clusters):
            cov = np.cov(cluster[:, point_relations[i]], rowvar=False)
            contours.append(create_contours(kmeans.cluster_centers_[j, point_relations[i]], cov))
        for j in range(k):
            ax.contour(contours[j][0], contours[j][1], contours[j][2], colors='black', alpha=contour_alpha)

    fig.suptitle(f"Digit {digit}")
    plt.tight_layout()
    plt.show()
