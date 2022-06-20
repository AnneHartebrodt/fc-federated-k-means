import time

from engine.app import AppState, app_state, Role

from apps.kmeans.config import FCConfig
from apps.kmeans.FCFederatedKMeans import FCFederatedKMeans
from apps.kmeans.COParams import COParams

# This is the first (initial) state all app instances are in at the beginning
# By calling it 'initial' the FeatureCloud template engine knows that this state is the first one to go into automatically at the beginning
@app_state('initial')  # The first argument is the name of the state ('initial'), the second specifies which roles are allowed to have this state (here BOTH)
class InitialState(AppState):

    def configure(self):
        print("[CLIENT] Parsing parameter file...", flush=True)
        if self.id is not None:  # Test is setup has happened already
            # parse parameters
            self.config = FCConfig()
            self.config.parse_configuration()
            self.store('configuration', self.config)
            print("[CLIENT] finished parsing parameter file.", flush=True)

    def register(self):
        self.register_transition('check_row_names', Role.COORDINATOR)
        self.register_transition('wait_for_params', Role.PARTICIPANT)

    def run(self):
        self.configure()
        print('[STARTUP] Instantiate SVD')
        if self.is_coordinator:
            self.store('kmeans', FCFederatedKMeans())
        else:
            self.store('kmeans', FCFederatedKMeans())
        self.load('kmeans').copy_configuration(self.config)
        print('[STARTUP] Configuration copied')

        # READ INPUT DATA
        self.load('kmeans').read_input_files()
        out = self.load('kmeans').out

        self.send_data_to_coordinator(out)

        if self.is_coordinator:
            return 'check_row_names'
        else:
            return 'wait_for_params'

@app_state('check_row_names', Role.COORDINATOR)
class CheckRowNames(AppState):
    '''
    This state collects all relevant parameters necessary to unify the run.
    Notably it makes sure the names of the variables match.

    '''
    def register(self):
        self.register_transition('wait_for_params', Role.COORDINATOR)


    def run(self):
        print('gathering')
        incoming = self.gather_data()
        print('unifying row names')
        self.load('kmeans').unify_row_names(incoming)
        out = self.load('kmeans').out
        self.broadcast_data(out)
        return 'wait_for_params'



@app_state('wait_for_params', Role.BOTH)
class WaitForParamsState(AppState):
    '''
    This state collects all relevant parameters necessary to unify the run.
    Notably it makes sure the names of the variables match.

    '''
    def register(self):
        self.register_transition('aggregate_sums', Role.COORDINATOR)
        self.register_transition('compute_std', Role.PARTICIPANT)
        self.register_transition('start_k_means', Role.BOTH)

    def run(self):
        incoming = self.await_data(is_json=False)
        print('setting parameters')
        self.load('kmeans').set_parameters(incoming)
        self.load('kmeans').select_rows(incoming)

        config = self.load('configuration')

        self.load('kmeans').compute_sums()
        out = self.load('kmeans').out
        self.send_data_to_coordinator(out)
        if self.is_coordinator:
            return 'aggregate_sums'
        else:
            return 'compute_std'


@app_state('aggregate_sums', Role.COORDINATOR)
class AggregateSummaryStatsState(AppState):
    def register(self):
        self.register_transition('compute_std', Role.COORDINATOR)

    def run(self):
        incoming = self.gather_data()
        print('setting parameters')
        self.load('kmeans').compute_means(incoming)
        out = self.load('kmeans').out
        self.broadcast_data(out)
        return 'compute_std'


@app_state('compute_std', Role.BOTH)
class ComputeSummaryStatsState(AppState):
    def register(self):
        self.register_transition('aggregate_stds', Role.COORDINATOR)
        self.register_transition('apply_scaling', Role.PARTICIPANT)

    def run(self):
        incoming = self.await_data()
        self.load('kmeans').compute_sum_of_squares(incoming)
        out = self.load('kmeans').out
        self.send_data_to_coordinator(out)
        if self.is_coordinator:
            return 'aggregate_stds'
        else:
            return 'apply_scaling'

@app_state('aggregate_stds', Role.COORDINATOR)
class AggregateStdState(AppState):
    def register(self):
        self.register_transition('apply_scaling', Role.COORDINATOR)

    def run(self):
        incoming = self.gather_data()
        print('setting parameters')
        self.load('kmeans').compute_std(incoming)
        out = self.load('kmeans').out
        self.broadcast_data(out)
        return 'apply_scaling'

@app_state('start_k_means', Role.BOTH)
class PowerIterationStartState(AppState):
    def register(self):
        self.register_transition('aggregate_centroids', Role.COORDINATOR)
        self.register_transition('update_centroids', Role.PARTICIPANT)

    def run(self):
        self.load('kmeans').init_kmeans()
        out = self.load('kmeans').out
        self.send_data_to_coordinator(out)

        if self.is_coordinator:
            return 'aggregate_centroids'
        else:
            return 'update_centroids'


@app_state('apply_scaling', Role.BOTH)
class ScaleDataState(AppState):

    def register(self):
        self.register_transition('start_k_means', Role.BOTH)


    def run(self):
        config = self.load('configuration')
        incoming = self.await_data()
        self.load('kmeans').apply_scaling(incoming, highly_variable=config.highly_variable)

        return 'start_k_means'


@app_state('aggregate_centroids', Role.COORDINATOR)
class AggregateCentroidsState(AppState):
    def register(self):
        self.register_transition('update_centroids', Role.COORDINATOR)

    def run(self):
        incoming = self.gather_data()
        self.load('kmeans').aggregate_centroids(incoming)
        out = self.load('kmeans').out
        self.broadcast_data(out)
        return 'update_centroids'


@app_state('update_centroids', Role.BOTH)
class UpdateCentroidsState(AppState):
    def register(self):
        self.register_transition('save_results', Role.BOTH)
        self.register_transition('aggregate_centroids', Role.COORDINATOR)
        self.register_transition('update_centroids', Role.COORDINATOR)

    def run(self) -> str:
        incoming = self.await_data()
        converged = self.load('kmeans').update_centroids(incoming)
        if converged:
            return 'save_results'
        else:
            if self.is_coordinator:
                return 'aggregate_centroids'
            else:
                return 'update_centroids'


@app_state('save_results')
class ShareProjectionsState(AppState):
    def register(self):
        self.register_transition('finalize', Role.BOTH)

    def run(self):
        print('SAVING RESULTS')
        config = self.load('configuration')

        self.load('kmeans').save_clustering()
        self.load('kmeans').save_logs()
        out = self.load('kmeans').out
        self.send_data_to_coordinator(out)
        return 'finalize'


@app_state('finalize')
class FinalizeState(AppState):
    def register(self):
        self.register_transition('terminal', Role.BOTH)

    def run(self):
        # Wait until all send the finished flag
        if self.is_coordinator:
            self.gather_data()
        return 'terminal'



