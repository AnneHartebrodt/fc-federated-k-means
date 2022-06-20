import os

import sklearn as sk
import numpy as np
import pandas as pd
import os.path as op
import argparse as ap
import sklearn.cluster as clu
import evaluation as eval

def compute_canonical(input_file, output_folder,k_min=3, k_max=10, k_step=1, seed = 11, prefix='singular', header=None, index_col=None,
                               sep='\t', center=True, variance=True, log=True):


    np.random.seed(seed)
    if isinstance(input_file, str):
        data = pd.read_csv(input_file, sep=sep, header=header, index_col=index_col)


    else:
        data = input_file

    rownames = data.index.values
    colnames =data.columns.values
    data = data.values



    if log:
        print('TRANSFORM')
        data = np.log2(data+1)


    nans = np.sum(np.isnan(data), axis=1)
    print(nans)
    nanfrac = nans / data.shape[1]
    nanindx = np.where(nanfrac <= 1)[0]
    print(nanindx)
    print(data)
    data = data[nanindx, :]

    means = np.nanmean(data, axis=1)
    means = np.atleast_2d(means).T

    if center:
        print('center')
        means = np.nanmean(data,  axis=1)
        means = np.atleast_2d(means).T
        data = data-means
        print(means)
        print(data)

    if variance:
        print('scale variance')
        print(np.nansum(np.square(data), axis=1))
        stds = np.nanstd(data, axis=1, ddof=1)
        stds = np.atleast_2d(stds).T
        print(stds)
        data = data/stds

    if center:
        print(data.shape)
        data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
    else:
        data = np.where(np.isnan(data), means, data)

    rownames = rownames[nanindx]

    print(np.cov(data))

    for k in range(k_min, k_max, k_step):
        kout = op.join(output_folder, 'K_'+str(k))
        print(kout)
        os.makedirs(kout, exist_ok=True)
        kmeans = clu.KMeans(n_clusters=k).fit(data)
        pd.DataFrame(kmeans.cluster_centers_).to_csv(op.join(kout, 'centroids.tsv'), sep=sep)
        pd.DataFrame(kmeans.labels_).to_csv(op.join(kout, 'clustering.tsv'), sep=sep)

        silhouette_scores = eval.simplified_silhouette_scores(data, kmeans.labels_, kmeans.cluster_centers_)
        pd.DataFrame(silhouette_scores).to_csv(op.join(kout, 'silhouette.tsv'), sep=sep)



if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Split complete data into test data for federated PCA')
    parser.add_argument('--directory', metavar='DIRECTORY', type=str, help='output directory', default='.')
    parser.add_argument('--output', metavar='OUTPUT_DIRECTORY_NAME', type=str, help='output directory', default='.')
    parser.add_argument('--filename', metavar='FILENAME', type=str, help='filename', default='data.tsv')

    parser.add_argument('--center', metavar='CENTER', type=bool, help='center matrices', default=False)
    parser.add_argument('--variance', metavar='VARIANCE', type=bool, help='scale matrices to unit variance',
                        default=False)
    parser.add_argument('--log_transform', metavar='LOG', type=bool, help='center matrices', default=False)
    parser.add_argument('--count', metavar='COUNTER', type=float, help='center matrices', default=None)
    parser.add_argument('--k_min', metavar='K MIN', type=int, help='minimal k to test', default=2)
    parser.add_argument('--k_max', metavar='K MAX', type=int, help='maximal k to test', default=15)
    parser.add_argument('--k_step', metavar='K STEP', type=int, help='K increment', default=1)
    parser.add_argument('--header', metavar='HEADER', type=int, help='header', default=None)
    parser.add_argument('--rownames', metavar='ROWNAMES', type=int, help='rownames', default=None)
    parser.add_argument('--seed', metavar='SEED', type=int, help='rownames', default=11)
    parser.add_argument('--delim', type=str, default='\t')

    args = parser.parse_args()
    basedir = args.directory
    datafile  = op.join(basedir, args.filename)
    output_folder = op.join(basedir, 'baseline', args.output)
    os.makedirs(output_folder, exist_ok=True)

    compute_canonical(datafile, output_folder=output_folder, k_min = args.k_min, k_max=args.k_max, k_step=args.k_step,
                      seed=11, header=args.header, index_col=args.rownames, sep=args.delim)
