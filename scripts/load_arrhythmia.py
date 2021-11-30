import scipy.io
from sklearn.model_selection import train_test_split

def load_arrhythmia(path, setting):

    data = scipy.io.loadmat(path)
    features = data["X"]
    labels = data["y"]

    pos_label = 0
    neg_label = 1
    algorithm = setting["z_run_name"]
    if (algorithm == "oneclass" or algorithm == "isolation" or algorithm == "covariance" or algorithm == "localoutlier"):
        labels[labels == neg_label] = -1
        labels[labels == pos_label] = 1

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)

    return X_train, y_train, X_test, y_test