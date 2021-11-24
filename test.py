from scripts.load_dataset import load_dataset
from scripts.min_max_scaler import min_max_scaler

setting = {
    "z_name_of_experiment": 'oneclass-arrhythmia',
    "z_run_name": "oneclass",
    "z_n_components": 1000,
    "z_tol": 1e-05, 
    "z_max_iter": 20000,
    "z_step": "train_val",
    "z_dataset": "arrhythmia",
    "z_test_running_times": 10
}

#prod_settings = {"z_gamma" : [2**i for i in range(-20,10)], "z_C": [2**i for i in range(-20,10)]}
prod_settings = {"z_gamma" : [2], "z_C": [2]}

params_int = ["z_n_components", "z_max_iter"]
params_float = ["z_tol","z_gamma", "z_C"]

mlflow = mlflow_create_experiment(setting["z_name_of_experiment"])

experiments(setting, prod_settings, params_int, params_float, mlflow)
