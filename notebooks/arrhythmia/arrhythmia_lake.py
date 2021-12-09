current_path = ""


try:  
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    import os
    import sys
    sys.path.append('submodules/qmc/')
    print(sys.path)
else:
    import sys
    sys.path.append('submodules/qmc/')
    sys.path.append('data/')
    print(sys.path)


print(os.getcwd())

sys.path.append('scripts/')


from mlflow_create_experiment import mlflow_create_experiment
from experiments import experiments

import torch


setting = {
    "z_name_of_experiment": 'lake-arrhythmia83',
    "z_run_name": "lake",
    "z_dataset": "arrhythmia", 
    "z_select_best_experiment": True,
    "z_batch_size": 64,
    "z_learning_rate": 1e-05,
    "z_iter_per_epoch": 100
}

prod_settings = {
    "z_ratio": [0.83]
}

params_int = ["z_batch_size","z_iter_per_epoch"]
params_float = ["z_learning_rate", "z_ratio"]

mlflow = mlflow_create_experiment(setting["z_name_of_experiment"])

experiments(setting, prod_settings, params_int, params_float, mlflow)