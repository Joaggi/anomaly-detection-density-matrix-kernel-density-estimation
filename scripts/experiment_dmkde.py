import qmc.tf.layers as layers
import qmc.tf.models as models

import numpy as np
from calculate_metrics import calculate_metrics
from find_best_threshold import find_best_threshold


def experiment_dmkde(X_train, y_train, X_test, y_test, settings, mlflow):

    for i, setting in enumerate(settings):
        #print(f"experiment_dmkdc {i} threshold {setting['z_threshold']}")
        with mlflow.start_run(run_name=setting["z_run_name"]):

            fm_x = layers.QFeatureMapRFF(X_train.shape[1], dim=setting["z_rff_components"], 
                                         gamma=setting["z_gamma"], random_state=setting["z_random_state"])
            qmd = models.QMDensity(fm_x, setting["z_rff_components"])
            qmd.compile()
            qmd.fit(X_train, epochs=1, batch_size=setting["z_batch_size"], verbose=0)
            
            y_test_pred = qmd.predict(X_test)

            if np.isclose(setting["z_threshold"], 0.0, rtol=0.0):
                thresh = find_best_threshold(y_test, y_test_pred)
                setting["z_threshold"] = thresh

            preds = (y_test_pred < setting["z_threshold"]).astype(int)
            metrics = calculate_metrics(y_test, preds)

            mlflow.log_params(setting)
            mlflow.log_metrics(metrics)

            print(f"experiment_dmkdc {i} metrics {metrics}")
            print(f"experiment_dmkdc {i} threshold {setting['z_threshold']}")