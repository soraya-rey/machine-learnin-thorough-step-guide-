import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


def split_data(input_path, label_col_index):     
    df = pd.read_csv(input_path)
    y = df.iloc[:, label_col_index]
    X = df.drop(df.columns[label_col_index], axis=1)    
    return X, y


def preprocess_labels(y):
    # Ensure y is 1D
    if isinstance(y, pd.DataFrame) or len(y.shape) > 1:
        y = y.squeeze()  # Convert to Series if it's a DataFrame
    
    # Use LabelEncoder for binary labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return y_encoded


def basic_imputation(X):
    imputers = []
    for column in X.columns:
        col_type = X[column].dtype
        if col_type == 'int64' or col_type == 'float64':  # Numerical columns
            imputers.append((f'num_imputer_{column}', SimpleImputer(strategy='median'), [column]))
        elif col_type == 'object':  # Categorical columns
            imputers.append((f'cat_imputer_{column}', SimpleImputer(strategy='most_frequent'), [column]))
    transformer = ColumnTransformer(imputers)
    X_imputed = transformer.fit_transform(X)
    return pd.DataFrame(X_imputed, columns=X.columns)


def advanced_imputation(X):
    imputers = []
    X = X.copy()
    
    # Handle numerical columns with regression
    for column in X.select_dtypes(include=['int64', 'float64']).columns:
        if X[column].isnull().any():
            missing = X[column].isnull()
            train = X.loc[~missing]
            test = X.loc[missing]
            
            reg_model = LinearRegression()
            reg_model.fit(train.drop(columns=[column]), train[column])
            X.loc[missing, column] = reg_model.predict(test.drop(columns=[column]))
    
    # Handle categorical columns with KNN
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        knn_imputer = KNNImputer(n_neighbors=5)
        X[categorical_cols] = knn_imputer.fit_transform(X[categorical_cols])
    
    return X


def evaluate_model(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if len(set(y)) > 2:  # Regression task
        return mean_squared_error(y_test, y_pred)
    else:  # Classification task
        y_pred_binary = (y_pred > 0.5).astype(int) if hasattr(model, "predict_proba") else y_pred
        return accuracy_score(y_test, y_pred_binary)


def compare_imputation_methods(file_dir):
    X,y = split_data(file_dir,1)
    y = preprocess_labels(y)
    # Define imputation methods
    imputation_methods = {
        "Basic Imputation (Median/Mode)": basic_imputation,
        "Advanced Imputation (Regression/KNN)": advanced_imputation
    }
    # Model to evaluate
    model = RandomForestClassifier()
    # Evaluate each method
    results = {}
    for method_name, impute_func in imputation_methods.items():
        X_imputed = impute_func(X)  
        results[method_name] = evaluate_model(X_imputed, y, model)
    # Display results
    print("Sensitivity Analysis Results:")
    for method, score in results.items():
        print(f"{method}: {score}")   
    return

def do_basic_imputation(file_dir,col_num):
    X,y = split_data(file_dir,col_num)
    y_encoded = preprocess_labels(y)
    X_imputed = basic_imputation(X)
    return X_imputed,y_encoded

def do_ml_imputation(file_dir,col_num):
    X,y = split_data(file_dir,col_num)
    y_encoded = preprocess_labels(y)
    X_imputed = advanced_imputation(X)
    return X_imputed,y_encoded


def delete_column(df, col_name):    
    if col_name in df.columns:
        df = df.drop(columns=[col_name])
    else:
        print(f"Column '{col_name}' does not exist in the DataFrame.")
    return df



