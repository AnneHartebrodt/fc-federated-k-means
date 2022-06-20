import yaml
import os.path as op
import os
import argparse as ap


def make_default_config_file(datafile = 'data.tsv',
                             center=True,
                             variance=True,
                             log_transform=True,
                             k_min=3,
                             k_max=10,
                             k_step=1,
                             output_dir='kmeans',
                             input_dir='',
                             delim = '\t'):

    """
    Default config file generator
    qr: one of 'federated_qr'| 'no_qr'
    :return:
    """
    dict = {'fc_kmeans':
             {'input':
                  {'dir': input_dir,
                    'file': datafile,
                   'delimiter': delim},
              'algorithm':
                  {
                      'k_min': k_min,
                      'k_max': k_max,
                      'k_step': k_step
                  },
              'output':
                  {'centroids': 'centroids.csv',
                   'clustering': 'clustering.csv',
                   'silhouette': 'silhouette.csv',
                   'dir': output_dir,
                   'delimiter': delim
                   },
              'scaling': {
                  'center': center,
                  'variance': variance,
                  'log_transform': log_transform,
              'max_nan_fraction': 1}
              }
            }
    return dict

def write_config(config, basedir, counter):
    os.makedirs(op.join(basedir,  str(counter)), exist_ok=True)
    with open(op.join(basedir,  str(counter), 'config.yaml'), 'w') as handle:
        yaml.safe_dump(config, handle, default_flow_style=False, allow_unicode=True)


def create_configs_power(output_folder,
                         datafile='data.tsv',
                         counter=0,
                         center=True,
                         variance=True,
                         log_transform=True,
                         k_min=3,
                         k_max=10,
                         k_step=1,
                         delim = '\t'
                         ):

    config = make_default_config_file(datafile=datafile,
                                          center=center,
                                          variance=variance,
                                          log_transform=log_transform,
                                        k_min=k_min,
                                        k_max=k_max,
                                        k_step=k_step,
                                      delim=delim
                                      )
    write_config(config=config, basedir=output_folder, counter=counter)
    counter = counter + 1
    return counter



if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Split complete data into test data for federated PCA')
    parser.add_argument('-d', metavar='DIRECTORY', type=str, help='output directory', default='.')
    parser.add_argument('-o', metavar='OUTPUT_DIRECTORY_NAME', type=str, help='output directory', default='.')
    parser.add_argument('-f', metavar='FILENAME', type=str, help='filename', default='data.tsv')

    parser.add_argument('--center', metavar='CENTER', type=bool, help='center matrices', default=False)
    parser.add_argument('--variance', metavar='VARIANCE', type=bool, help='scale matrices to unit variance', default=False)
    parser.add_argument('--log_transform', metavar='LOG', type=bool, help='center matrices', default=False)
    parser.add_argument('--count', metavar='COUNTER', type=float, help='center matrices', default=None)
    parser.add_argument('--k_min', metavar='K MIN', type=int, help='minimal k to test',default=2)
    parser.add_argument('--k_max', metavar='K MAX', type=int, help='maximal k to test', default=15)
    parser.add_argument('--k_step', metavar='K STEP', type=int, help='K increment', default=1)
    parser.add_argument('--delim', type=str, default='\t')

    args = parser.parse_args()
    basedir = args.d

    output_folder = op.join(basedir, args.o)
    os.makedirs(output_folder, exist_ok=True)



    if args.count is None:
        count = 0
    else:
        count = args.count

    print(args.log_transform)
    count = create_configs_power(output_folder,
                                 datafile=args.f,
                                 counter=count,
                                 center=args.center,
                                 variance=args.variance,
                                 log_transform=args.log_transform,
                                 k_min =args.k_min,
                                 k_max = args.k_max,
                                 k_step=args.k_step,
                                 delim = args.delim
                                 )


