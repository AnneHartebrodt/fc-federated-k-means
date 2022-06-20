import json
import os

import pandas as pd
import copy
import sklearn.cluster as clu
from numpy.random import RandomState
import numpy as np
import os.path as op
from apps.kmeans.params import *

class Clustering:
    def __init__(self, centroids,  tabdata, k_min, k_max, k_step):
        self.k_min = k_min
        self.k_max = k_max
        self.k_spte = k_step
        self.tabdata = tabdata
        self.k_list = list(range(k_min, k_max + 1))
        self.centroids = centroids
        self.rs = RandomState(11)
        self.silhouette_scores = {}
        self.silhouette_coefficient = {}
        self.labels = {}



    @classmethod
    def init_local_clustering(cls,  tabdata, k_min, k_max, k_step) -> 'Clustering':
        """
        Initialise an SVD object with random matrices
        :param k: number of eigenvectors, (default: 10)
        :param tabdata: TabData object.
        :return:
        """
        cls.rs = RandomState(11)
        k_list = list(range(k_min, k_max + 1))
        centroids = {}
        for k in k_list:
            clud = clu.KMeans(n_clusters=k, n_init=50, random_state=cls.rs)
            clud.fit(tabdata.data)
            print(clud.cluster_centers_.shape)
            centroids[k] = clud.cluster_centers_
        return cls(centroids, tabdata, k_min, k_max, k_step)

    def aggregate_centroids(self, centroid_dict):
        for k in centroid_dict:
            centroids = np.concatenate(centroid_dict[k], axis=0)
            clud = clu.KMeans(n_clusters=k, n_init=50, random_state=self.rs)
            clud = clud.fit(centroids)
            self.centroids[k] = clud.cluster_centers_

    def update_local_centroid(self, global_centroids, converged= False):
        """
        Basically run one round of k-means with preset centroids
        :param data: local data set
        :param init_centoids: current centroid estimate
        :return: updated centroid estimate
        """
        print('Updating local centroids')
        self.centroids = global_centroids
        print(global_centroids)
        for k in global_centroids:
            clu_all = clu.KMeans(n_clusters=k, n_init=1,
                             random_state=self.rs, init=global_centroids[k],
                             max_iter=1)
            clu_all.fit(self.tabdata.data)
            self.labels[k] = clu_all.labels_
            if converged:
                self.silhouette_scores[k] = self.simplified_silhouette_scores(clu_all.labels_, clu_all.cluster_centers_)
                self.silhouette_coefficient[k] = np.nanmean(self.silhouette_scores[k])

        return clu_all.cluster_centers_



    def simplified_silhouette_scores(self,  label, centroids):
        # cannot use sklearn silhouette coefficient here,
        # because it relies solely on the labels, which means, the centroids
        # will be computed in a wrong manner.

        # need to compute simpliefied silhouette coefficient based on the
        # global centroids

        if len(np.unique(label)) <= 1:
            return [0] * self.tabdata.data.shape[0]

        nearest_centroid = []
        nearest_distance = []

        second_best_centroid = []
        second_distance = []

        for i in range(self.tabdata.data.shape[0]):
            # compute second best centroid for each data point
            current_nearest = 0
            current_nearest_distance = np.inf

            current_second = 1
            current_second_distance = np.inf

            for c in range(centroids.shape[0]):
                # compute distance
                cd = np.linalg.norm(self.tabdata.data[i, :] - centroids[c, :])
                if cd < current_nearest_distance:
                    if current_nearest_distance < current_second_distance:
                        current_second_distance = current_nearest_distance
                        current_second = current_nearest
                    current_nearest_distance = cd
                    current_nearest = c
                elif cd < current_second_distance:
                    current_second_distance = cd
                    current_second = c

            nearest_centroid.append(current_nearest)
            nearest_distance.append(current_nearest_distance)
            second_best_centroid.append(current_second)
            second_distance.append(current_second_distance)

        silhouette_value = []
        for i in range(self.tabdata.data.shape[0]):
            try:
                sv = (second_distance[i] - nearest_distance[i]) / max(second_distance[i], nearest_distance[i])
            except:
                sv = -1
            silhouette_value.append(sv)

        # set silhouette for singletons to 0
        bc = np.bincount(nearest_centroid)
        for i in np.where(bc == 1)[0]:
            silhouette_value[np.where(nearest_centroid == i)[0][0]] = 0

        return silhouette_value

    def to_tsv(self, centroids_file, clustering_file, silhouette_file, delim='\t'):
        for k in self.k_list:
            cen = op.join(OUTPUT_DIR, 'kmeans', 'K_' + str(k), centroids_file)
            clu = op.join(OUTPUT_DIR, 'kmeans', 'K_' + str(k), clustering_file)
            silh =op.join(OUTPUT_DIR, 'kmeans', 'K_' + str(k), silhouette_file)

            print('output dir')
            print(cen)
            os.makedirs(op.join(OUTPUT_DIR, 'kmeans', 'K_' + str(k)))
            print(op.join(OUTPUT_DIR, 'kmeans', 'K_' + str(k)))
            pd.DataFrame(self.centroids[k]).to_csv(cen, sep=delim)
            pd.DataFrame(self.labels[k]).to_csv(clu, sep=delim)
            pd.DataFrame(self.silhouette_scores[k]).to_csv(silh, sep=delim)
