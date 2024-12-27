from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

def normalize_data(X):
    """
    Normalizes the dataset X to have values between 0 and 1 for each feature.
    Args:
    - X (DataFrame): Features dataset.
    
    Returns:
    - X_normalized (DataFrame): Normalized features dataset.
    """
    # Initialize MinMaxScaler (scales each feature to [0, 1])
    scaler = MinMaxScaler()
    
    # Apply normalization
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    print("Normalization applied: Features scaled between 0 and 1")
    return X_normalized

def standardize_data(X):
    """
    Standardizes the dataset X to have mean 0 and standard deviation 1 for each feature.
    Args:
    - X (DataFrame): Features dataset.
    
    Returns:
    - X_standardized (DataFrame): Standardized features dataset.
    """
    # Initialize StandardScaler (standardizes features to have mean=0 and std=1)
    scaler = StandardScaler()
    
    # Apply standardization
    X_standardized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    print("Standardization applied: Features scaled to mean 0 and std 1")
    return X_standardized
