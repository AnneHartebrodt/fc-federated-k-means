import yaml
import os
import re
import os.path as op
from apps.kmeans.params import INPUT_DIR, OUTPUT_DIR
from shutil import copyfile


class FCConfig:
    def __init__(self):

        self.config_available = False
        self.input_file = None

        self.input_dir = None

        self.scaled_data_file = None
        self.k_min = 2
        self.k_max = 15
        self.k_step=1
        self.sep = '\t'
        self.has_rownames = 0
        self.has_colnames = 0
        self.encryption = False
        self.use_smpc = False
        self.exponent = 3

        self.output_dir = None
        self.output_delim = '\t'
        self.centroids_file = None
        self.clustering_file = None
        self.silhouette_file = None

        self.center = True
        self.unit_variance = True
        self.highly_variable = True
        self.perc_highly_var = 0.1
        self.log_transform = True
        self.max_nan_fraction = 0.5

    def parse_configuration(self):
        print('[API] /setup parsing parameter file ')
        regex = re.compile('^config.*\.(yaml||yml)$')
        config_file = "ginger_tea.txt"
        # check input dir for config file
        files = os.listdir(INPUT_DIR)
        for file in files:
            if regex.match(file):
                config_file = op.join(INPUT_DIR, file)
                config_out = op.join(OUTPUT_DIR, file)
                break
        # check output dir for config file
        files = os.listdir(OUTPUT_DIR)
        for file in files:
            if regex.match(file):
                config_file = op.join(OUTPUT_DIR, file)
                break
        if op.exists(config_file):
            # Copy file to output folder
            print('[API] /setup config file found ... parsing file: ' + str(op.join(INPUT_DIR, config_file)))
            copyfile(config_file, config_out)

            self.config_available = True
            with open(config_file, 'r') as file:
                parameter_list = yaml.safe_load(file)
                parameter_list = parameter_list['fc_kmeans']

                print(parameter_list)
                # Files
                try:
                    self.input_dir = parameter_list['input']['dir']
                    self.input_file = parameter_list['input']['file']

                except KeyError:
                    print('YAML file does not follow specification: missing key '+ str('data'))
                    raise KeyError

                try:
                    self.sep = parameter_list['input']['delimiter']
                except KeyError:
                    print('YAML file does not follow specification: delimiter not specified')
                    raise KeyError

                try:
                    self.centroids_file =  parameter_list['output']['centroids']
                    self.clustering_file = parameter_list['output']['clustering']
                    self.silhouette_file = parameter_list['output']['silhouette']
                    self.output_dir = parameter_list['output']['dir']
                    self.output_delim = parameter_list['output']['delimiter']

                except KeyError:
                    self.clustering_file = 'clustering.tsv'
                    self.silhouette_file = 'silhouette.tsv'
                    self.centroids_file = 'centroids.tsv'

                try:
                    self.scaled_data_file =   parameter_list['output']['scaled_data_file']
                except KeyError:
                    print('YAML file does not follow specification: missing key: projections')
                    print('Setting default: projections.tsv')
                    self.scaled_data_file = 'scaled_data.tsv'

                try:
                    self.k_min = parameter_list['algorithm']['k_min']
                except KeyError:
                    print('K MIN not specified, defaulting to 2')
                    self.k_min = 2

                try:
                    self.k_max = parameter_list['algorithm']['k_max']
                except KeyError:
                    print('K MAX not specified, defaulting to 15')
                    self.k_max = 15

                try:
                    self.k_step = parameter_list['algorithm']['k_step']
                except KeyError:
                    print('K MAX not specified, defaulting to 1')
                    self.k_step =1



                try:
                    self.center = parameter_list['scaling']['center']
                    self.unit_variance = parameter_list['scaling']['variance']
                    self.log_transform = parameter_list['scaling']['log_transform']
                    self.max_nan_fraction = parameter_list['scaling']['max_nan_fraction']
                except KeyError:
                    print('Scaling functionalities not specified.')

                try:
                    #self.use_smpc = parameter_list['privacy']['use_smpc']
                    #self.exponent = parameter_list['privacy']['exponent']
                    self.use_smpc=False
                    self.exponent = 3

                except KeyError:
                    print('YAML file does not follow specification: privacy settings')
                    raise KeyError

                print('[API] /setup config file found ... parsing done')

        else:
            print('[API] /setup no configuration file found')
            self.config_available = False