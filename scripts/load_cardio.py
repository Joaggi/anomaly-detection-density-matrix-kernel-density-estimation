import numpy as np
from sklearn.model_selection import train_test_split

def load_cardio(path):

    data = np.load(path)
    features = data[:,:-1]
    labels = data[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)

    return X_train, y_train, X_test, y_test