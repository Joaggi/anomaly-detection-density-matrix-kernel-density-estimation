from sklearn.metrics import accuracy_score, f1_score

def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics
