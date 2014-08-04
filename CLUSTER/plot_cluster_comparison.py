"""
=========================================================
Comparing different clustering algorithms on toy datasets
=========================================================

This example aims at showing characteristics of different
clustering algorithms on datasets that are "interesting"
but still in 2D. The last dataset is an example of a 'null'
situation for clustering: the data is homogeneous, and
there is no good clustering.

While these examples give some intuition about the algorithms,
this intuition might not apply to very high dimensional data.

The results could be improved by tweaking the parameters for
each clustering strategy, for instance setting the number of
clusters for the methods that needs this parameter
specified. Note that affinity propagation has a tendency to
create many clusters. Thus in this example its two parameters
(damping and per-point preference) were set to to mitigate this
behavior.
@author: http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
@date: July 21, 2014 (retrieved from above)
@modified: Nelson & Hobbes modifications to suit personal use
"""
print(__doc__)


# ##################################################################################
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
# ##################################################################################


# ##################################################################################
N_SAMPLES = 1500
# ##################################################################################


# ##################################################################################
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ##################################################################################


# ##################################################################################
def load_dataset( dname, num_samples ):
    if 'circles' in dname.lower():
        noisy_circles = datasets.make_circles(n_samples=num_samples, factor=.5, noise=.05)
        return noisy_circles
    elif 'moons' in dname.lower():
        noisy_moons = datasets.make_moons(n_samples=num_samples, noise=.05)
        return noisy_moons
    elif 'blobs' in dname.lower():
        blobs = datasets.make_blobs(n_samples=num_samples, random_state=8)
        return blobs
    else:
        no_structure = np.random.rand(num_samples, 2), None
        return no_structure
    return [[]]
# ##################################################################################


# ##################################################################################
def plot_clustering_results( X, y_pred, algorithm, timediff=0.0 ):
    COLORS = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    COLORS = np.hstack([COLORS] * 20)

    plt.scatter(X[:, 0], X[:, 1], color=COLORS[y_pred].tolist(), s=10)

    if hasattr(algorithm, 'cluster_centers_'):
        centers = algorithm.cluster_centers_
        center_colors = COLORS[:len(centers)]
        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xticks(())
    plt.yticks(())

    plt.text(.99, .01, ('%.2fs' % (timediff)).lstrip('0'), transform=plt.gca().transAxes, size=15, horizontalalignment='right')

    return
# ##################################################################################


# ##################################################################################
def analysis():
    np.random.seed(0)

    DATASETS = [ load_dataset('noisy_circles', N_SAMPLES),
                 load_dataset('noisy_moons', N_SAMPLES),
                 load_dataset('blobs', N_SAMPLES),
                 load_dataset('no_structure', N_SAMPLES) ]

    plt.figure(figsize=(17, 9.5))
    plt.subplots_adjust(left=.001, right=.999, bottom=.001, top=.96, wspace=.05, hspace=.01)

    PLOT_NUM = 1
    for i_dataset, dataset in enumerate(DATASETS):
        X, y = dataset

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(X, n_neighbors=10)

        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # Compute distances, #distances = np.exp(-euclidean_distances(X))
        distances = euclidean_distances(X)

        # create clustering estimators
        # ward            = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward', connectivity=connectivity)
        # average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=2, connectivity=connectivity)
        kmeans               = cluster.KMeans(n_clusters=2, init='k-means++', n_init=10, precompute_distances=True, copy_x=True)
        two_means            = cluster.MiniBatchKMeans(n_clusters=2)
        affinity_propagation = cluster.AffinityPropagation(damping=.9, preference=-200)
        ms                   = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        spectral             = cluster.SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity="nearest_neighbors")
        dbscan               = cluster.DBSCAN(eps=.2)

        ALGORITHMS = [          # ('Ward', ward),
                                # ('AgglomerativeClustering', average_linkage),
                                ('BasicKmeans', kmeans),
                                ('MiniBatchKMeans', two_means),
                                ('AffinityPropagation', affinity_propagation),
                                ('MeanShift', ms),
                                ('SpectralClustering', spectral),
                                ('DBSCAN', dbscan) ]

        for name, algorithm in ALGORITHMS:
            # predict cluster memberships
            t0 = time.time()
            algorithm.fit(X)
            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)

            # plot
            plt.subplot(len(DATASETS), len(ALGORITHMS), PLOT_NUM)

            plot_clustering_results( X, y_pred, algorithm, t1-t0)

            if i_dataset == 0:
                plt.title(name, size=18)

            PLOT_NUM += 1

    plt.show()


# ##################################################################################
if __name__ == "__main__":

    analysis()
