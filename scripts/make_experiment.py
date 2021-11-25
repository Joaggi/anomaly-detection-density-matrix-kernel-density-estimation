from experiment_oneclass import experiment_oneclass
from experiment_isolation import experiment_isolation

def make_experiment(algorithm, X_train, y_train, X_test, y_test, settings, mlflow):
    
    if algorithm == "oneclass":
        experiment_oneclass(X_train, y_train, X_test, y_test, settings, mlflow)
    if algorithm == "isolation":
        experiment_isolation(X_train, y_train, X_test, y_test, settings, mlflow)