from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import h5py

def XGBoost_Model(y_train = None):
    """
    Function to create XGBClassifier Model, pass y_train if DATA is unbalanced
    """
    
    if y_train is not None:
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        
        return XGBClassifier(scale_pos_weight=scale_pos_weight)
        
    else:
        return XGBClassifier()
        
        
def save_model_h5(model, file_path):
    """
    Save the XGBoost model to an HDF5 (.h5) file.
    
    Args:
        model (XGBClassifier): The trained XGBoost model.
        file_path (str): Path to save the model (e.g., "model.h5").
    """
    # Save the model in .h5 format
    model_booster = model.get_booster()
    model_binary = model_booster.save_raw()
    with h5py.File(file_path, "w") as h5_file:
        h5_file.create_dataset("xgboost_model", data=np.void(model_binary))
    
    

def load_model_h5(file_path):
    """
    Load an XGBoost model from an HDF5 (.h5) file.
    
    Args:
        file_path (str): Path to the saved model file (e.g., "model.h5").
    
    Returns:
        XGBClassifier: The loaded XGBoost model.
    """
    return joblib.load(file_path)
    
    
def Train_model(model):
    model.fit(x_train, y_train)
    pre = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]  # Get probabilities for the positive class

    accuracy = accuracy_score(y_test, pre)
    recall = recall_score(y_test, pre)
    f1 = f1_score(y_test, pre)
    auc = roc_auc_score(y_test, proba)

    print(f"Accuracy: {accuracy}    Recall: {recall}    F1: {f1}    AUC: {auc}")
    
    
def train_and_evaluate_xgboost(model ,X_train, y_train, X_test, y_test):
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=False)
    print("\nClassification Report:")
    print(report)
    
    # ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {roc_auc}")
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()