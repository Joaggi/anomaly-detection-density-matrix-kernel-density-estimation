import numpy as np
from sklearn.model_selection import train_test_split

def load_kdd(path):

    data = np.load(path, allow_pickle=True)
    features = data["kdd"][:,:-1]
    labels = data["kdd"][:,-1]

    inds = np.random.choice(len(features), size=50000, replace=False)

    X_train, X_test, y_train, y_test = train_test_split(features[inds], labels[inds], test_size=0.3, random_state=42, stratify=labels[inds])

    return X_train, y_train, X_test, y_test