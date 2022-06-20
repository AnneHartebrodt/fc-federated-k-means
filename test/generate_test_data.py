import numpy as np
import pandas as pd
import scipy as sc
import sklearn as sk
from sklearn.datasets import make_blobs
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
import os.path as op
import os
import argparse as ap


def make_test_data(basedir, filename= "blobs_varicance", centers = 15, number_points_per_center=500,
                   n_features=2, l_variance=[1.0], outlier_percentage=0.01, alpha=2,
                   box=5, delim='\t', filesuffix='.tsv'):
    '''

    :param basedir: Name of the output directory
    :param centers: Number of centers to generate,
    :param number_points_per_center:  Number of points per center, remember to scale up
                the number of points if the data is to be split into federated data.
    :param n_features: number of dimensions of the data
    :return:
    '''
    # generate the same data but with different cluster separation
    # aka variance between the clusters.
    # we increased the bounding box from standard settings to allow for
    # proper cluster separation
    # for convenience the center boundaries are set to +/- the number of
    # clusters

    total_points = centers * number_points_per_center


    for cluster_variance in l_variance:
        direc = op.join(basedir, str(cluster_variance))

        os.makedirs(direc, exist_ok=True)

        X, y, centers = make_blobs(n_samples=total_points, n_features=n_features, centers=centers, cluster_std=cluster_variance,
                               center_box=(-centers-5, centers+5), return_centers=True)

        if outlier_percentage is not None or outlier_percentage!=0:
            rng = np.random.default_rng()
            rng.random()

            random_points = []
            nr_outliers = int(np.floor(outlier_percentage * total_points))
            for d in range(centers.shape[1]):
                minim = np.min(centers[:, 1]) - box
                maxim = np.max(centers[:, 1]) + box
                random_points.append((maxim - minim) * rng.random(nr_outliers) + minim)

            random_points = np.stack(random_points, axis=1)

            reject_points = []
            i = 0
            for rp in random_points:
                for c in centers:
                    if np.linalg.norm(rp - c) < cluster_variance * alpha:
                        reject_points.append(i)
                        break
                i = i + 1

            random_points = np.delete(random_points, reject_points, axis=0)
            y_out = [-1] * random_points.shape[0]
            X = np.concatenate([random_points, X])
            y = np.concatenate([y_out, y])


        # save the data into a csv sheet
        print(cluster_variance)
        file_prefix = filename
        print(op.join(direc, file_prefix+filesuffix))
        pd.DataFrame(X).to_csv(op.join(direc, file_prefix+filesuffix), sep=delim, header=True, index=True)

        # save the labels into a csv sheet
        pd.DataFrame(y).to_csv(op.join(direc, file_prefix+'.labels'+filesuffix), sep=delim, header=True, index=True)

        #also save a plot for your convienience.
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y)
        plt.savefig(op.join(direc, file_prefix+'.pdf'))
        plt.close()

    with open(op.join(direc, filename + '.log'), 'w') as handle:
        handle.write("done")




if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--directory', type=str)
    parser.add_argument('--variances', type=float, nargs='+')
    parser.add_argument('--nfeatures', type=int)
    parser.add_argument('--points', type=int)
    parser.add_argument('--centers', type=int)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--outlier_percentage', type=float, default=0.0)
    parser.add_argument('--delim', type=str, default='\t')
    parser.add_argument('--filesuffix', type=str, default='.tsv')

    args = parser.parse_args()


    # this data directory will be created
    #directory = '/home/anne/Documents/featurecloud/kmeans/data/simulated_test_data'

    print(args.variances)
    # these are the default settings
    make_test_data(args.directory, filename=args.filename, centers = args.centers, number_points_per_center=args.points,
    n_features=args.nfeatures, l_variance=args.variances, outlier_percentage = args.outlier_percentage, delim=args.delim, filesuffix=args.filesuffix)


    # total_points = 300
    # n_features = 2
    # centers = 5
    # cluster_variance = 1
    # box = 5
    # outlier_percentage = 0.1
    # alpha = 2








