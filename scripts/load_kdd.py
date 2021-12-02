import numpy as np
from sklearn.model_selection import train_test_split

def load_kdd(path, setting):

    data = np.load(path, allow_pickle=True)
    features = data["kdd"][:,:-1]
    labels = data["kdd"][:,-1]

    _, sub_train, _, sub_test = train_test_split(features, labels, test_size=5000, random_state=42, stratify=labels)

    pos_label = 1
    algorithm = setting["z_run_name"]
    y = []
    if (algorithm == "oneclass" or algorithm == "isolation" or algorithm == "covariance" or algorithm == "localoutlier"):
        for el in sub_test:
            y.append(1 if el==pos_label else -1)

    X_train, X_test, y_train, y_test = train_test_split(sub_train, y, test_size=0.3, random_state=42, stratify=y)

    return X_train, y_train, X_test, y_test