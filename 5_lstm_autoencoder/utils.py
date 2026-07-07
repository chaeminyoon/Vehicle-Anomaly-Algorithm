# utils.py
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt

def evaluate_model(model, test_data, true_labels, threshold):
    reconstructions = model.predict(test_data)
    mse = np.mean(np.square(test_data - reconstructions), axis=(1,2))
    
    predicted_anomalies = mse > threshold
    
    # PR curve
    precision, recall, _ = precision_recall_curve(true_labels, mse)
    pr_auc = auc(recall, precision)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(true_labels, mse)
    roc_auc = auc(fpr, tpr)
    
    print(f"PR AUC: {pr_auc}")
    print(f"ROC AUC: {roc_auc}")
    
    # Plot curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return predicted_anomalies, mse
