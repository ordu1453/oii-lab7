import numpy as np
import multiprocessing

from mst_clustering.clustering_models import ZahnModel, GathGevaModel
from sklearn.datasets import make_blobs
from mst_clustering import Pipeline


if __name__ == "__main__":
    multiprocessing.freeze_support()

    X, y = make_blobs(n_samples=1000, n_features=10, centers=7)

    clustering = Pipeline(clustering_models=[
        ZahnModel(3, 1.5, 1e-4, max_num_of_clusters=7),GathGevaModel(0.0001, 2)
    ])
    clustering.fit(data=X, workers_count=4)

    labels = clustering.labels
    partition = clustering.partition
    clusters_count = clustering.clusters_count

    print(labels)
    print(partition)
    print(clusters_count)