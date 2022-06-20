import sys
sys.path.append('/home/anne/Documents/featurecloud/apps/fc-federated-svd/')


import pandas as pd
import argparse as ap
import numpy as np
import os as os
import os.path as op
import yaml
import markdown
import test.markdown_utils as md
import test.evaluation as ev


def read_and_concatenate_eigenvectors(file_list):
    eigenvector_list=[]
    for f in file_list:
        eig0 = pd.read_csv(f, sep='\t', index_col=args.rownames, header=args.header)
        eig0 = eig0.values
        eigenvector_list.append(eig0)
    eig = np.concatenate(eigenvector_list, axis=0)
    return eig

def read_config(configfile):
    with open(op.join(configfile), 'r') as handle:
        config = yaml.safe_load(handle)
    return config

def read_iterations(iteration_file):
    with open(iteration_file, 'r') as handle:
        iterations = handle.readline().split()[1]
        runtime = round(float(handle.readline().split()[1]), 2)
        print(iterations)
        print(runtime)
    return iterations, runtime




def create_result(left_angles, right_angles, diff, config, run_id='NA', config_path='NA', result_path='NA'):
    l = []
    names = []
    names.append('Run ID')

    ar = run_id.split('_')
    l.append(ar[0].split('.')[0])


    for key in config:
        names.append(key)
        l.append(config[key])
    for a in range(len(left_angles)):
        names.append('LSV'+str(a+1))
        l.append(left_angles[a])
    for a in range(len(right_angles)):
        names.append('RSV'+str(a+1))
        l.append(right_angles[a])
    for d in range(len(diff)):
        names.append('SV'+str(d+1))
        l.append(diff[d])
    data = pd.DataFrame(l).T
    data.columns = names
    return data




if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Split complete data into test data for federated PCA')
    parser.add_argument('--baseline', metavar='DIRECTORY', type=str, help='output directory', default='.')
    parser.add_argument('--federated', metavar='DIRECTORY', type=str, help='output directory', default='.')

    parser.add_argument('-o', metavar='OUTPUT', type=str, help='filename of evaluation output')
    parser.add_argument('-e', metavar='CONFIG', type=str, help='config file')
    parser.add_argument('-i', metavar='ITERATIONS', type=str, help='iteration file')
    parser.add_argument('--header', metavar='HEADER', type=int, help='header (line number)', default=None)
    parser.add_argument('--rownames', metavar='ROW NAMES', type=int, help='row names (column number)', default=None)
    args = parser.parse_args()
    basedir = args.baseline

    k_runs = list(os.walk(basedir))[0][1]

    for k in k_runs:
        baseline_results = pd.fread(op.join(basedir, k, 'clustering.csv'), header=0, rownames=0)
        federated_result = pd.fread(op.join(args.federated, k, 'clustering.csv'), header=0, rownames=0)
        matching, gl_labels = ev.find_cluster_matching(baseline_results.iloc[:,0], federated_result.iloc[:,0])
        # log_labelling(filename=filename, clustering_scheme=lab + '-' + mode, repeat=r, labels=gl_labels, run_id=run_id)
        gl_ma_f1, gl_f1, gl_precision, gl_recall = ev.evaluate(baseline_results.iloc[:,0], gl_labels)
        #silhouette = ev.simplified_silhouette_coefficient(data, gl_labels, global_centroids)





    config = read_config(configfile=args.e)
    subconf = config['fc_pca']['algorithm']
    #subconf['smpc'] = config['fc_pca']['privacy']['use_smpc']
    subconf['center'] = config['fc_pca']['scaling']['center']
    subconf['variance'] = config['fc_pca']['scaling']['variance']
    subconf['log_transform'] = config['fc_pca']['scaling']['log_transform']


    subconf['iterations'], subconf['runtime'] = read_iterations(args.i)
    #subconf['runtime'] = read_iterations(args.i)[1]

    # ouput_table = create_result(gl_ma_f1, gl_f1, gl_precision, gl_recall,  subconf,
    #                             run_id=args.o,
    #                             config_path=args.e,
    #                             result_path = args.R
    #                             )
    # ouput_table.to_csv(op.join(op.join(basedir, 'test_results', args.o)), sep='\t', index=False)
    #
    #
