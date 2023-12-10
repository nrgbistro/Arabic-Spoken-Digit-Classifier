import numpy as np
from matplotlib import pyplot as plt
from GMM import GaussianMixtureModel
from parsing.ParsedData import ParsedData
from parsing.dataParser import parse_file
from scipy.stats import multivariate_normal


def generate_models():
    hyperparameters = {
        "mfcc_indexes": [i for i in range(13)],
        "use_kmeans": False,
        "covariance_type": "full",
        "covariance_tied": False,
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
    training_data = ParsedData(parse_file("spoken_arabic_digits/Train_Arabic_Digit.txt", 66)).filter_by_digit(0)
    GMM_sph = GaussianMixtureModel(training_data, hyperparameters)
    em_sph = GMM_sph.train_gmm(0)
    hyperparameters["covariance_tied"] = True
    GMM_diag = GaussianMixtureModel(training_data, hyperparameters)
    em_diag = GMM_diag.train_gmm(0)
    all_mfccs = np.vstack([data_block.mfccs for data_block in training_data.filter_by_digit(0)])
    return em_sph, em_diag, all_mfccs


def plot_clusters(cluster_params, mfccs, ax, title):
    size = 0.75
    point_alpha = 0.5

    v0, v1 = mfccs[:, :2].T
    points = [v0, v1]

    ax.scatter(points[1], points[0], c=cluster_params[0]["labels"], s=size,
               alpha=point_alpha)
    ax.set_title(title)

    return ax


def plot_level_sets(mean, cov, ax, x_extent, y_extent):
    # Create a grid of points within the extent of the data
    x = np.linspace(x_extent[0], x_extent[1], 100)
    y = np.linspace(y_extent[0], y_extent[1], 100)
    X, Y = np.meshgrid(x, y)

    # Stack the points into a 2D array
    pos = np.dstack((X, Y))

    # Extract the appropriate indices for the mean and covariance
    # Assuming you want to plot the first and second MFCC (index 0 and 1)
    mean_plot = mean[[1, 0]]  # Swap indices if needed
    cov_plot = cov[np.ix_([1, 0], [1, 0])]  # Swap rows and columns if needed

    # Create the multivariate normal distribution
    rv = multivariate_normal(mean_plot, cov_plot)

    # Calculate the PDF at each point
    Z = rv.pdf(pos)

    # Plot the level sets
    ax.contour(X, Y, Z)


def compare_em_kmeans(tied, distinct, all_mfccs):
    x_extent = (np.min(all_mfccs[:, 1]), np.max(all_mfccs[:, 1]))
    y_extent = (np.min(all_mfccs[:, 0]), np.max(all_mfccs[:, 0]))
    fig, (sph_ax, diag_ax) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    for cluster in tied:
        plot_level_sets(cluster["mean"], cluster["covariance"], sph_ax, x_extent, y_extent)
    plot_clusters(tied, all_mfccs, sph_ax, "Distinct Covariance")
    sph_ax.set_aspect('equal', 'box')

    for cluster in distinct:
        plot_level_sets(cluster["mean"], cluster["covariance"], diag_ax, x_extent, y_extent)
    plot_clusters(distinct, all_mfccs, diag_ax, "Tied Covariance")
    diag_ax.set_aspect('equal', 'box')

    fig.suptitle("Comparing Tied and Distinct Covariance")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    tiedd, distinctt, MFCCs = generate_models()
    compare_em_kmeans(tiedd, distinctt, MFCCs)
