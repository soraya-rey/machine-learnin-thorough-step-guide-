from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier, BalancedBaggingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.svm import OneClassSVM

# Random Over-Sampling
def random_over_sampling(X, y):
    """
    Performs random over-sampling to balance the dataset.
    
    Args:
        X: Feature matrix.
        y: Target variable.
    
    Returns:
        X_resampled, y_resampled: Resampled feature matrix and target.
    """
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled

# Random Under-Sampling
def random_under_sampling(X, y):
    """
    Performs random under-sampling to balance the dataset.
    
    Args:
        X: Feature matrix.
        y: Target variable.
    
    Returns:
        X_resampled, y_resampled: Resampled feature matrix and target.
    """
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled

# SMOTE (Synthetic Minority Over-sampling Technique)
def smote_sampling(X, y):
    """
    Performs SMOTE to generate synthetic samples for the minority class.
    
    Args:
        X: Feature matrix.
        y: Target variable.
    
    Returns:
        X_resampled, y_resampled: Resampled feature matrix and target.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# ADASYN (Adaptive Synthetic Sampling)
def adasyn_sampling(X, y):
    """
    Performs ADASYN to generate adaptive synthetic samples for the minority class.
    
    Args:
        X: Feature matrix.
        y: Target variable.
    
    Returns:
        X_resampled, y_resampled: Resampled feature matrix and target.
    """
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    return X_resampled, y_resampled

# Logistic Regression with Class Weights Adjustment
def logistic_regression_balanced(X, y):
    """
    Trains a Logistic Regression model with balanced class weights.
    
    Args:
        X: Feature matrix.
        y: Target variable.
    
    Returns:
        model: Trained Logistic Regression model.
    """
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X, y)
    return model

# Balanced Random Forest
def balanced_random_forest(X, y):
    """
    Trains a Balanced Random Forest model to handle imbalanced data.
    
    Args:
        X: Feature matrix.
        y: Target variable.
    
    Returns:
        model: Trained Balanced Random Forest model.
    """
    model = BalancedRandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

# XGBoost with Scale_Pos_Weight for Imbalance Handling
def xgboost_balanced(X, y, ratio_of_classes):
    """
    Trains an XGBoost model with scale_pos_weight to handle class imbalance.
    
    Args:
        X: Feature matrix.
        y: Target variable.
        ratio_of_classes: Ratio between majority and minority classes.
    
    Returns:
        model: Trained XGBoost model.
    """
    model = xgb.XGBClassifier(scale_pos_weight=ratio_of_classes, random_state=42)
    model.fit(X, y)
    return model

# One-Class SVM for Anomaly Detection
def one_class_svm(X):
    """
    Trains a One-Class SVM model to detect anomalies (outliers).
    
    Args:
        X: Feature matrix.
    
    Returns:
        model: Trained One-Class SVM model.
    """
    model = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
    model.fit(X)
    return model

# EasyEnsemble
def easy_ensemble(X, y):
    """
    Trains an EasyEnsemble model to handle imbalanced datasets.
    
    Args:
        X: Feature matrix.
        y: Target variable.
    
    Returns:
        model: Trained EasyEnsemble model.
    """
    model = EasyEnsembleClassifier(random_state=42)
    model.fit(X, y)
    return model

# Balanced Bagging Classifier
def balanced_bagging(X, y):
    """
    Trains a Balanced Bagging Classifier to address class imbalance.
    
    Args:
        X: Feature matrix.
        y: Target variable.
    
    Returns:
        model: Trained Balanced Bagging model.
    """
    model = BalancedBaggingClassifier(random_state=42)
    model.fit(X, y)
    return model
