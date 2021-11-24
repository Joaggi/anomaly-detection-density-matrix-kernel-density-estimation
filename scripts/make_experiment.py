from experiment_oneclass import experiment_oneclass



def make_experiment(algorithm, X_train, y_train, X_test, y_test, settings, mlflow):
    
    if algorithm == "oneclass":
        experiment_oneclass(X_train, y_train, X_test, y_test, settings, mlflow)