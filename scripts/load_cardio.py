import numpy as np
from sklearn.model_selection import train_test_split

def load_cardio(path, algorithm):

    data = np.load(path)
    features = data[:,:-1]
    labels = data[:,-1]

    pos_label = 1

    y = []
    if (algorithm == "oneclass" or algorithm == "isolation" or algorithm == "covariance" or algorithm == "localoutlier"):
        for el in labels:
            y.append(1 if el==pos_label else -1)
    elif (algorithm.startswith("dmkde") or algorithm == "lake" or algorithm.startswith("pyod")):
        for el in labels:
            y.append(0 if el==pos_label else 1)

    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42, stratify=y)

    return X_train, y_train, X_test, y_test