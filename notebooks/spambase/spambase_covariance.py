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
    "z_name_of_experiment": 'covariance-spambase',
    "z_run_name": "covariance",
    "z_dataset": "spambase", 
    "z_labels": [1,-1]
}

#prod_settings = {"z_gamma" : [2**i for i in range(-20,10)], "z_C": [2**i for i in range(-20,10)]}
prod_settings = {"z_nu": [0.1]}

params_int = []
params_float = ["z_nu"]

mlflow = mlflow_create_experiment(setting["z_name_of_experiment"])

experiments(setting, prod_settings, params_int, params_float, mlflow)
