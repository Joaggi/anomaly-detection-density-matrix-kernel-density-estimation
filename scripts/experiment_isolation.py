import numpy as np
from sklearn.ensemble import IsolationForest

from calculate_metrics import calculate_metrics


def experiment_isolation(X_train, y_train, X_test, y_test, settings, mlflow):
    
    for i, setting in enumerate(settings):

        #print(f"experiment_dmkdc {i} setting {setting}")
        with mlflow.start_run(run_name=setting["z_run_name"]):

            model = IsolationForest(n_estimators=setting["z_n_estimators"], max_samples=setting["z_max_samples"], 
                                    contamination=setting["z_nu"], random_state=setting["z_random_state"])
            model.fit(X_train)
            y_test_pred = model.predict(X_test)

            metrics = calculate_metrics(y_test, y_test_pred)

            mlflow.log_params(setting)
            mlflow.log_metrics(metrics)
            print(f"experiment_dmkdc {i} metrics {metrics}")