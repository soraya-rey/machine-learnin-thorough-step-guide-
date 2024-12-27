from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
    RFE,
    SequentialFeatureSelector,
)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector
from xgboost import XGBClassifier
import shap

def chi2_selection(X, y, k=10):
    selector = SelectKBest(chi2, k=k)
    selector.fit(X, y)
    X_optimal = selector.transform(X)
    rankings = selector.scores_
    return X_optimal, rankings

def f_classif_selection(X, y, k=10):
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)
    X_optimal = selector.transform(X)
    rankings = selector.scores_
    return X_optimal, rankings

def mutual_info_selection(X, y, k=10):
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X, y)
    X_optimal = selector.transform(X)
    rankings = selector.scores_
    return X_optimal, rankings

def rfe_selection(X, y, n_features_to_select=10):
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    estimator = LogisticRegression(solver='liblinear')
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    selector.fit(X_normalized, y)
    X_optimal = selector.transform(X_normalized)
    rankings = selector.ranking_
    return X_optimal, rankings

def forward_selection(X, y, n_features_to_select=10):
    estimator = LogisticRegression(solver='liblinear')
    selector = SequentialFeatureSelector(estimator, n_features_to_select=n_features_to_select, direction='forward', scoring='accuracy', cv=5)
    selector.fit(X, y)
    X_optimal = selector.transform(X)
    rankings = selector.get_support(indices=True)
    return X_optimal, rankings

def backward_selection(X, y, n_features_to_select=10):
    estimator = LogisticRegression(solver='liblinear')
    selector = SequentialFeatureSelector(estimator, n_features_to_select=n_features_to_select, direction='backward', scoring='accuracy', cv=5)
    selector.fit(X, y)
    X_optimal = selector.transform(X)
    rankings = selector.get_support(indices=True)
    return X_optimal, rankings

def lasso_selection(X, y, alpha=0.1):
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_normalized, y)
    rankings = np.abs(lasso.coef_)
    selected_features = np.where(lasso.coef_ != 0)[0]
    X_optimal = X.iloc[:, selected_features]
    return X_optimal, rankings

def tree_based_selection(X, y, n_features_to_select=10):
    forest = RandomForestClassifier(random_state=42)
    forest.fit(X, y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    selected_features = indices[:n_features_to_select]
    X_optimal = X.iloc[:, selected_features]
    return X_optimal, importances

def extra_trees_selection(X, y, n_features_to_select=10):
    trees = ExtraTreesClassifier(random_state=42)
    trees.fit(X, y)
    importances = trees.feature_importances_
    indices = np.argsort(importances)[::-1]
    selected_features = indices[:n_features_to_select]
    X_optimal = X.iloc[:, selected_features]
    return X_optimal, importances

def gradient_boosting_selection(X, y, n_features_to_select=10):
    gboost = GradientBoostingClassifier(random_state=42)
    gboost.fit(X, y)
    importances = gboost.feature_importances_
    indices = np.argsort(importances)[::-1]
    selected_features = indices[:n_features_to_select]
    X_optimal = X.iloc[:, selected_features]
    return X_optimal, importances

def pca_selection(X, n_components=10):
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_optimal = pca.fit_transform(X_normalized)
    rankings = pca.explained_variance_ratio_
    return X_optimal, rankings

def spearman_selection(X, y, n_components=10):    
    # Spearman correlation between each feature and the target
    correlations = []
    for feature in X.columns:
        corr, _ = spearmanr(X[feature], y)
        correlations.append(corr)    
    # Rank the features based on their correlation values
    rankings = np.argsort(np.abs(correlations))[::-1]  # Sort by absolute value of correlation    
    # Get top k features based on Spearman correlation
    top_k_features = rankings[:n_components]
    X_optimal = X.iloc[:, top_k_features]    
    return X_optimal, correlations

def shap_selection(X, y, n_components=10, model=None):
    # Standardize the features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)    
    # Default model is RandomForestClassifier
    if model is None:
        model = RandomForestClassifier(random_state=42)    
    # Train the model
    model.fit(X_normalized, y)    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_normalized)    
    # For classification, shap_values is a list of arrays (one for each class), take the mean absolute value of the SHAP values
    shap_importance = np.mean(np.abs(shap_values[1]), axis=0)  # Assuming binary classification (index 1 for the positive class)
    # Get top k features based on SHAP values
    top_k_features = np.argsort(shap_importance)[::-1][:n_components]
    X_optimal = X.iloc[:, top_k_features]    
    return X_optimal, shap_importance



def compare_methods_numerical(X, y):
    """
    Applies feature selection methods compatible with numerical data.

    Args:
        X: Feature matrix (DataFrame with numerical data).
        y: Target variable (1D array or Series).
        feature_names: List of feature names.

    Returns:
        Dictionary of feature rankings across methods.
    """
    # Check input
    # Updated check to allow any float or integer types
    if not np.issubdtype(X.dtypes[0], np.floating) and not np.issubdtype(X.dtypes[0], np.integer):
        raise ValueError("X must contain numerical data (either integers or floats) for these methods.")
    if not np.issubdtype(y.dtype, np.floating) and not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y must be numerical (either integers or floats) for these methods.")


    feature_rankings = {}
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # ANOVA F-test
    f_classif_selector = SelectKBest(f_classif, k='all')
    f_classif_selector.fit(X, y)
    feature_rankings["ANOVA F-test"] = f_classif_selector.scores_

    # Mutual Information
    mutual_info_selector = SelectKBest(mutual_info_classif, k='all')
    mutual_info_selector.fit(X, y)
    feature_rankings["Mutual Info"] = mutual_info_selector.scores_

    # Lasso
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_normalized, y)
    feature_rankings["Lasso"] = np.abs(lasso.coef_)

    # Random Forest
    forest = RandomForestClassifier(random_state=42)
    forest.fit(X, y)
    feature_rankings["Random Forest"] = forest.feature_importances_

    # Spearman Correlation
    spearman_scores = [spearmanr(X.iloc[:, i], y).correlation for i in range(X.shape[1])]
    feature_rankings["Spearman Correlation"] = np.abs(spearman_scores)

    return feature_rankings


from sklearn.feature_selection import mutual_info_classif

def compare_methods_categorical(X, y):
    """
    Applies feature selection methods compatible with categorical data.

    Args:
        X: Feature matrix (DataFrame with categorical or non-negative numerical data).
        y: Target variable (1D array or Series).

    Returns:
        Dictionary of feature rankings across methods.
    """
    # Ensure X is categorical or non-negative numerical
    if not all([pd.api.types.is_categorical_dtype(X[col]) or np.issubdtype(X[col].dtype, np.number) for col in X.columns]):
        raise ValueError("X must contain categorical or non-negative numerical data for these methods.")

    # Encode y if categorical
    if pd.api.types.is_categorical_dtype(y):
        y = LabelEncoder().fit_transform(y)

    feature_rankings = {}

    # Chi2 (requires non-negative integers)
    X_non_negative = X.apply(lambda col: col - col.min() if col.min() < 0 else col)
    chi2_selector = SelectKBest(chi2, k='all')
    chi2_selector.fit(X_non_negative, y)
    feature_rankings["Chi2"] = chi2_selector.scores_

    # Information Gain (Mutual Information)
    mutual_info_selector = SelectKBest(mutual_info_classif, k='all')
    mutual_info_selector.fit(X_non_negative, y)
    feature_rankings["Information Gain"] = mutual_info_selector.scores_

    return feature_rankings


def compare_methods_universal(X, y):
    """
    Applies feature selection methods compatible with both numerical and categorical data.

    Args:
        X: Feature matrix (DataFrame with numerical or categorical data).
        y: Target variable (1D array or Series).
        feature_names: List of feature names.

    Returns:
        Dictionary of feature rankings across methods.
    """
    feature_rankings = {}

    # RFE
    rfe_model = LogisticRegression(solver="liblinear")
    rfe_selector = RFE(estimator=rfe_model, n_features_to_select=1, step=1)
    rfe_selector.fit(X, y)
    feature_rankings["RFE"] = -rfe_selector.ranking_  # Negative ranking to match importance scale

    # Forward Selection
    forward_selector = SequentialFeatureSelector(
        LogisticRegression(solver="liblinear"), n_features_to_select=1, direction="forward"
    )
    forward_selector.fit(X, y)
    feature_rankings["Forward Selection"] = -np.argsort(forward_selector.get_support())[::-1]

    # Backward Selection
    backward_selector = SequentialFeatureSelector(
        LogisticRegression(solver="liblinear"), n_features_to_select=1, direction="backward"
    )
    backward_selector.fit(X, y)
    feature_rankings["Backward Selection"] = -np.argsort(backward_selector.get_support())[::-1]

    # Tree-Boosted Method (XGBoost Feature Importances)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    xgb_model.fit(X, y)
    feature_rankings["Tree-Boosted (XGBoost)"] = xgb_model.feature_importances_

    # SHAP (SHapley Additive exPlanations)
    explainer = shap.Explainer(xgb_model, X)
    shap_values = explainer(X)
    shap_importances = np.abs(shap_values.values).mean(axis=0)
    feature_rankings["SHAP"] = shap_importances

    return feature_rankings


def plot_feature_importance_comparison(feature_rankings, feature_names):
    """
    Plots a heatmap comparing feature rankings across different methods.

    Args:
        feature_rankings: Dictionary where keys are method names and values are arrays of feature rankings.
        feature_names: List of feature names.
    """
    # Convert to DataFrame for heatmap
    ranking_df = pd.DataFrame(feature_rankings, index=feature_names)

    # Normalize rankings to [0, 1] for comparability
    ranking_df = (ranking_df - ranking_df.min()) / (ranking_df.max() - ranking_df.min())

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        ranking_df,
        annot=True,
        cmap="viridis",
        cbar_kws={'label': 'Normalized Importance'},
        linewidths=0.5,
    )
    plt.title("Comparison of Feature Rankings Across Methods")
    plt.xlabel("Feature Selection Methods")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()


def plot_method_similarity(feature_rankings):
    """
    Plots a heatmap of correlation between feature selection methods.

    Args:
        feature_rankings: Dictionary where keys are method names and values are arrays of feature rankings.
    """
    # Convert rankings to DataFrame
    ranking_df = pd.DataFrame(feature_rankings)

    # Compute correlation matrix
    corr_matrix = ranking_df.corr()

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        cbar_kws={'label': 'Correlation'},
        linewidths=0.5,
    )
    plt.title("Correlation Between Feature Selection Methods")
    plt.tight_layout()
    plt.show()
