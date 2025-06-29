from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm, columns=['Predicted: 0', 'Predicted: 1'], index=['Actual: 0', 'Actual: 1']
    )
    plt.figure(figsize=(7, 4))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_roc(clf, X_test, y_test):
    probs = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    fpr, tpr, _ = roc_curve(y_test, probs)
    
    plt.figure(figsize=(8, 5))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {round(auc, 3)})')
    plt.tight_layout()
    plt.show()
    return auc