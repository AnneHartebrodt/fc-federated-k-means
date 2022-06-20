import os
import os.path as op
from apps.kmeans.TabData import TabData
import pandas as pd
import traceback
import numpy as np
import copy
from apps.kmeans.params import INPUT_DIR, OUTPUT_DIR
from apps.kmeans.COParams import COParams
import time
from apps.kmeans.Clustering import Clustering
from apps.kmeans.serialize import *
import shutil

class FCFederatedKMeans:
    def __init__(self):
        self.step = 0
        self.tabdata = None
        self.clustering = None
        self.config_available = False
        self.out = None
        self.send_data = False
        self.computation_done = False
        self.coordinator = False
        self.iteration_counter = 0
        self.converged = False
        self.data_incoming = {}
        self.progress = 0.0
        self.silent_step=False
        self.use_smpc = False
        self.start_time = time.monotonic()
        self.means = None
        self.std = None
        self.total_sampels = 0


    def copy_configuration(self, config):
        print('[STARTUP] Copy configuration')
        self.config_available = config.config_available

        self.centroids_file =  config.centroids_file
        self.clustering_file = config.clustering_file
        self.silhouette_file = config.silhouette_file
        self.output_dir =  op.join(OUTPUT_DIR, config.output_dir)
        self.output_delim = config.output_delim

        self.input_file = op.join(INPUT_DIR, config.input_dir, config.input_file)
        self.means_file = op.join(OUTPUT_DIR, config.output_dir, 'mean.tsv')
        self.stds_file = op.join(OUTPUT_DIR, config.output_dir, 'std.tsv')
        self.log_file = op.join(OUTPUT_DIR, config.output_dir, 'run_log.txt')

        self.input_dir = config.input_dir
        self.k_min = config.k_min
        self.k_max = config.k_max
        self.k_step = config.k_step

        self.exponent = config.exponent
        self.sep = config.sep
        self.has_rownames = config.has_rownames
        self.has_colnames = config.has_colnames

        self.center = config.center
        self.unit_variance = config.unit_variance
        self.log_transform = config.log_transform
        self.max_nan_fraction = config.max_nan_fraction


    def read_input_files(self):
        self.progress = 0.1
        self.tabdata = TabData.from_file(self.input_file, header=self.has_colnames,
                                         index=self.has_rownames, sep=self.sep)

        if self.log_transform:
            print('Log Transform performed')
            self.tabdata.scaled = np.log2(self.tabdata.scaled+1)

        nans = np.sum(np.isnan(self.tabdata.scaled), axis=1)
        infs = np.sum(np.isinf(self.tabdata.scaled), axis=1)
        isneginf = np.sum(np.isneginf(self.tabdata.scaled), axis=1)
        nans = np.sum([nans, isneginf, infs], axis=0)
        self.out = {COParams.ROW_NAMES.n : self.tabdata.rows, COParams.SAMPLE_COUNT.n: self.tabdata.col_count, COParams.NAN.n: nans}



    def set_parameters(self, incoming):
        print('[API] Setting parameters')
        self.k_min = incoming[COParams.KMIN.n]
        self.k_max = incoming[COParams.KMAX.n]
        self.k_step = incoming[COParams.KSTEP.n]


    def select_rows(self, incoming):
        subset = incoming[COParams.ROW_NAMES.n]
        print(subset)
        d = {k: v for v, k in enumerate(self.tabdata.rows)}
        index = []
        for elem in subset:
            if elem in d:
                index.append(d[elem])
        print('INDEX')
        print(index)
        self.tabdata.scaled = self.tabdata.scaled[index,:]
        self.tabdata.rows = self.tabdata.rows[index]
        self.tabdata.row_count = len(self.tabdata.rows)

    def unify_row_names(self, incoming):
        '''
        Make sure the clients use a set of common row names.
        Make sure the maximal fraction of NAs is not exceeded.

        Parameters
        ----------
        incoming Incoming data object from clients

        Returns
        -------

        '''
        print(incoming)
        mysample_count = 0
        myintersect = set(incoming[0][COParams.ROW_NAMES.n])

        nandict = {}
        for s in incoming:
            for n, v in zip(s[COParams.ROW_NAMES.n], s[COParams.NAN.n]):
                if n in nandict:
                    nandict[n] = nandict[n]+v
                else:
                    nandict[n] = v
            myintersect = myintersect.intersection(set(s[COParams.ROW_NAMES.n]))
            mysample_count = s[COParams.SAMPLE_COUNT.n]+mysample_count

        select = []
        for n in nandict:
            fract = nandict[n]/mysample_count
            if fract<=self.max_nan_fraction:
                select.append(n)

        print(select)
        myintersect = myintersect.intersection(set(select))
        self.total_sampels = mysample_count
        self.out = {COParams.KMIN.n: self.k_min, COParams.KMAX.n: self.k_max, COParams.KSTEP.n: self.k_step}
        newrownames = list(myintersect)
        self.out[COParams.ROW_NAMES.n] = newrownames

        values_per_row = []
        for n in newrownames:
            values_per_row.append(mysample_count-nandict[n])
        self.values_per_row = values_per_row

        print('[API] [COORDINATOR] row names identified!')


    def compute_sums(self):
        self.sums = np.nansum(self.tabdata.scaled, axis=1)

        self.out = {COParams.SUMS.n: self.sums, COParams.SAMPLE_COUNT.n: self.tabdata.col_count}

    def compute_sum_of_squares(self, incoming):
        self.means = incoming[COParams.MEANS.n].reshape((len(incoming[COParams.MEANS.n]),1))
        print(self.means.shape)
        self.sos = np.nansum(np.square(self.tabdata.scaled-self.means), axis=1)
        self.out = {COParams.SUM_OF_SQUARES.n: self.sos.flatten()}

    def apply_scaling(self, incoming, highly_variable=True):
        self.std = incoming[COParams.STDS.n].reshape((len(incoming[COParams.STDS.n]),1))
        remove = incoming[COParams.REMOVE.n] # remove due to 0
        select = incoming[COParams.SELECT.n] # select due to highly var
        # for row in range(self.tabdata.scaled.shape[0]):
        #     self.tabdata.scaled[row, :]= self.tabdata.scaled[row, :]- self.means[row,0]
        if self.center:
            self.tabdata.scaled = np.subtract(self.tabdata.scaled,self.means)


        # self.tabdata.scaled = np.delete(self.tabdata.scaled, remove)
        # self.tabdata.rows = np.delete(self.tabdata.rows, remove)
        if self.unit_variance:
            self.tabdata.scaled = self.tabdata.scaled/self.std

        if self.center:
            # impute. After centering, the mean should be 0, so this effectively mean imputation
            self.tabdata.scaled = np.nan_to_num(self.tabdata.scaled, nan=0, posinf=0, neginf=0)
        else:
            # impute
            self.tabdata.scaled = np.where(np.isnan(self.tabdata.scaled), self.means, self.tabdata.scaled)

        return self.tabdata.scaled.shape[0]

    def compute_means(self, incoming):
        print(incoming)
        my_sums = []
        my_samples = 0

        for s in incoming:
            my_sums.append(s[COParams.SUMS.n])
            my_samples = my_samples+s[COParams.SAMPLE_COUNT.n]

        my_sums = np.stack(my_sums)
        my_sums = np.nansum(my_sums, axis=0)

        my_sums = my_sums/self.values_per_row
        print('SUMS')
        print(my_sums)

        self.out = {COParams.MEANS.n : my_sums }
        self.number_of_samples = my_samples

    def compute_std(self, incoming):
        my_ssq  = []
        for s in incoming:
            print(s[COParams.SUM_OF_SQUARES.n])
            my_ssq.append(s[COParams.SUM_OF_SQUARES.n])
        my_ssq = np.stack(my_ssq)
        my_ssq = np.nansum(my_ssq, axis=0)
        print('COMPUTE STD')
        print(my_ssq)
        val_per_row = [v-1 for v in self.values_per_row]
        my_ssq = np.sqrt(my_ssq/(val_per_row))
        self.std = my_ssq
        print('STD')
        print(self.std)

        hv = self.tabdata.scaled.shape[0]


        remove = np.where(self.std.flatten()==0)
        # std in fact contains the standard deviation
        select = np.argsort(self.std.flatten())[0:hv]

        REM = self.tabdata.rows[remove]
        SEL = self.tabdata.rows[select]


        self.out = {COParams.STDS.n : self.std, COParams.SELECT.n: select, COParams.REMOVE.n: remove}


    def init_kmeans(self):
        self.clustering = Clustering.init_local_clustering(self.tabdata, self.k_min, self.k_max, self.k_step)
        self.out = {COParams.CENTROIDS.n: serialize_dict(self.clustering.centroids)}
        print(self.out)


    def update_centroids(self, incoming):
        converged = False
        print('Updating')
        if incoming[COParams.CONVERGED.n]:
            converged = True
            print(incoming[COParams.CENTROIDS.n])
            self.clustering.update_local_centroid(deserialize_dict(incoming[COParams.CENTROIDS.n]), converged)
        return converged

    def aggregate_centroids(self, incoming):
        converged = True
        print(incoming)

        # initialize empty dictionary
        local_centroids = {}
        for k in self.clustering.k_list:
            local_centroids[k] = []

        # iterate over all clients (list of dicts)
        for obj in incoming:
            # iterate over all k
            # dict key = k, value = centroids (serialized numpy arrays)
            for k, v in obj[COParams.CENTROIDS.n].items():
                print(v)
                local_centroids[k].append(deserialize_array(v))

        self.out = {COParams.CENTROIDS.n: serialize_dict(self.clustering.centroids),COParams.CONVERGED.n: converged}




    def save_clustering(self):
        # update PCA and save
        os.makedirs(self.output_dir)
        self.copy_input_to_output()
        self.save_logs()
        if self.means is not None:
            pd.DataFrame(self.means).to_csv(self.means_file, sep=self.output_delim)
            pd.DataFrame(self.std).to_csv(self.stds_file, sep=self.output_delim)

        self.clustering.to_tsv(self.centroids_file, self.clustering_file, self.silhouette_file, delim=self.output_delim)


    def save_logs(self):
        with open(self.log_file, 'w') as handle:
            handle.write('iterations:\t'+str(self.iteration_counter)+'\n')
            handle.write('runtime:\t' + str(time.monotonic()-self.start_time)+'\n')
        self.out = {COParams.FINISHED: True}


    def copy_input_to_output(self):
        print('MOVE INPUT TO OUTPUT')
        shutil.copytree(INPUT_DIR, OUTPUT_DIR, dirs_exist_ok=True)
