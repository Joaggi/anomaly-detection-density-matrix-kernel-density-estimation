current_path = ""


try:  
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    import os
    os.system("pip3 install mlflow")
    from google.colab import drive
    drive.mount('/content/drive')
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

import tensorflow as tf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from experiments import experiments

setting = {
    "z_name_of_experiment": 'covariance-cardio',
    "z_run_name": "covariance",
    "z_dataset": "cardio", 
    "z_pos_label": 1,
    "z_neg_label": -1,
    "z_select_best_experiment": True
}

prod_settings = {
    "z_nu": [i/100 for i in range(1,41)]
}

params_int = ["z_pos_label", "z_neg_label"]
params_float = ["z_nu"]

mlflow = mlflow_create_experiment(setting["z_name_of_experiment"])

experiments(setting, prod_settings, params_int, params_float, mlflow)