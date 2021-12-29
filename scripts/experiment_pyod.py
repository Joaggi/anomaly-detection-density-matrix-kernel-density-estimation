from pyod.models.knn import KNN
from pyod.models.sos import SOS
from pyod.models.copod import COPOD
from pyod.models.xgbod import XGBOD
from pyod.models.vae import VAE
from pyod.models.deep_svdd import DeepSVDD

import numpy as np
from calculate_metrics import calculate_metrics


def experiment_pyod(X_train, y_train, X_test, y_test, settings, mlflow):

    for i, setting in enumerate(settings):
        #print(f"experiment_dmkdc {i} threshold {setting['z_threshold']}")
        with mlflow.start_run(run_name=setting["z_run_name"]):

            model = None

            if (setting["z_run_name"] == "pyod-knn"):
                model = KNN(contamination=setting["z_nu"], n_neighbors=setting["z_n_neighbors"])
                
            if (setting["z_run_name"] == "pyod-sos"):
                model = SOS(contamination=setting["z_nu"], perplexity=setting["z_perplexity"],
                            eps=setting["z_tol"])

            if (setting["z_run_name"] == "pyod-copod"):
                model = COPOD(contamination=setting["z_nu"])

            if (setting["z_run_name"] == "pyod-xgbod"):
                model = XGBOD(contamination=setting["z_nu"], random_state=setting["z_random_state"])

            if (setting["z_run_name"] == "pyod-vae"):
                model = VAE(contamination=setting["z_nu"], random_state=setting["z_random_state"],
                            batch_size=setting["z_batch_size"], epochs=setting["z_epochs"], verbose=0)

            if (setting["z_run_name"] == "pyod-deepsvdd"):
                model = DeepSVDD(contamination=setting["z_nu"], random_state=setting["z_random_state"], 
                                 epochs=setting["z_epochs"], verbose=0)


            if (setting["z_run_name"] == "pyod-xgbod"):
                model.fit(X_train, y_train)
            else:
                model.fit(X_train)


            y_test_pred = model.predict(X_test)

            metrics = calculate_metrics(y_test, y_test_pred)

            mlflow.log_params(setting)
            mlflow.log_metrics(metrics)

            print(f"experiment_pyod {i} metrics {metrics}")
