from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def calculate_metrics(y_true, y_pred, y_scores, run_name):

    auroc = roc_auc_score(y_true, y_scores)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average='weighted'),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "auroc": ((1-auroc) if (run_name=="lake" or run_name.startswith("dmkde")) else auroc)
    }
    
    return metrics
