import missing_values_imputation as impute
import exploratory_data_analysis as eda
import standardization as std 
import feature_selection as fs
import balancing_classes as balance
import build_models as build


file_dir = "./data/breast_cancer.csv"

# STEP 1 : handle missing values

X,y = impute.do_basic_imputation(file_dir,col_num=1)

# STEP * : manually delete columns obselete

X  = impute.delete_column(X, col_name='id')

# STEP 2 : EDA

# what are the data types in the datatset ?
eda.unique_data_types(X)
# are the features correlated ?
eda.clusters_heatmap(X)
# are the features distributed a certain way ?
eda.distribution_analysis(X)
eda.distribution_analysis_of_repartition(X)
# are the classes balanced ?
eda.plot_label_distribution(y)


# STEP 3 : standardization 

X_normalized = std.normalize_data(X)
X_scaled = std.standardize_data(X)

# STEP 4 : feature selection 

feature_rank_dict_universal = fs.compare_methods_universal(X,y)
feature_rank_dict_numerical = fs.compare_methods_numerical(X,y)
feature_names = X.columns
fs.plot_feature_importance_comparison(feature_rank_dict_universal,feature_names)
fs.plot_feature_importance_comparison(feature_rank_dict_numerical,feature_names)

X_gradient_boosting, feat_imp = fs.gradient_boosting_selection(X,y)
X_lasso, ranking = fs.lasso_selection(X,y)
X_rfe, ranks = fs.rfe_selection(X,y,n_features_to_select=15)
X_spearman , impttce = fs.spearman_selection(X,y,n_components=15)
X_shap , rank = fs.shap_selection(X,y,n_components=15)

# STEP 5 : handle data class imbalence 

# Example: Using one of the functions to balance the dataset
X_resampled, y_resampled = balance.smote_sampling(X, y)
# Example: Training a model with the balanced dataset
model_LG = balance.logistic_regression_balanced(X_resampled, y_resampled)
# Or, training with an ensemble method
model_EE = balance.easy_ensemble(X_resampled, y_resampled)


# STEP 6 : build machine learning models

build.compare_models(X,y) 
build.compare_models(X_normalized,y)
build.compare_linear_vs_nonlinear_models(X,y)
build.compare_linear_vs_nonlinear_models(X_normalized,y)

# List of feature selection methods

datasets_dict = {
    "Gradient Boosting": (X_gradient_boosting, y),
    "Lasso": (X_lasso, y),
    "RFE": (X_rfe, y),
    "Spearman": (X_spearman, y),
    "SHAP": (X_shap, y)
}

build.compare_models_with_feature_selection(datasets_dict)