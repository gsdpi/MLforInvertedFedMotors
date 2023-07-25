import os

path_here = os.path.abspath('')
dataset_dir = str(path_here)+'/Datasets/'
results_grid_search = str(path_here)+'/Results/Params/'
path_results_metrics = str(path_here)+'/Results/'

def get_param_path(modelID):
    return os.path.join(results_grid_search,modelID+'.csv')
