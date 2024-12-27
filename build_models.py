import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score



# Define models as functions for reuse
def logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return model, y_pred_prob

def random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return model, y_pred_prob

def gradient_boosting(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return model, y_pred_prob

def adaboost(X_train, y_train, X_test, y_test):
    model = AdaBoostClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return model, y_pred_prob

def svc(X_train, y_train, X_test, y_test):
    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return model, y_pred_prob

def knn(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return model, y_pred_prob

def naive_bayes(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return model, y_pred_prob



def compare_models(X, y):
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # List of models
    models = {
        "Logistic Regression": logistic_regression,
        "Random Forest": random_forest,
        "Gradient Boosting": gradient_boosting,
        "AdaBoost": adaboost,
        "SVC": svc,
        "KNN": knn,
        "Naive Bayes": naive_bayes
    }

    # Store the results for plotting
    model_names = []
    auc_scores = []
    cross_val_results = []
    fpr_tpr_auc = []  # Store fpr, tpr, and auc for each model to plot later

    # Evaluate models
    for name, model_func in models.items():
        print(f"{name} Evaluation:")
        model, y_pred_prob = model_func(X_train, y_train, X_test, y_test)

        # Cross-validation score (accuracy)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cross_val_results.append(cv_scores)

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)
        model_names.append(name)

        # Save fpr, tpr, and auc for plotting later
        fpr_tpr_auc.append((name, fpr, tpr, auc_score))

        # Print classification report
        print(classification_report(y_test, model.predict(X_test)))

    # Cross-validation boxplot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=cross_val_results)
    plt.xticks(np.arange(len(models)), model_names, rotation=45)
    plt.title('Cross-validation Accuracy Scores')
    plt.ylabel('Accuracy')

    # ROC curve plot (all models)
    plt.subplot(1, 2, 2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line for random classifier
    for name, fpr, tpr, auc_score in fpr_tpr_auc:
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

    # Finalize plots
    plt.title('ROC Curve Comparison')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.legend(loc="lower right")  # Add legend with model name and AUC for each model
    plt.show()

    # Return cross-validation results and AUC scores
    return cross_val_results, auc_scores



def compare_linear_vs_nonlinear_models(X, y):
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # List of models - separating linear models from non-linear models
    linear_models = {
        "Logistic Regression": logistic_regression,
        "Linear SVC": svc  # Linear SVC is typically linear
    }

    nonlinear_models = {
        "Random Forest": random_forest,
        "Gradient Boosting": gradient_boosting,
        "AdaBoost": adaboost,
        "KNN": knn,
        "Naive Bayes": naive_bayes
    }

    # Store the results for plotting
    model_names = []
    auc_scores = []
    fpr_tpr_auc = []  # Store fpr, tpr, and auc for each model to plot later

    # Evaluate linear models
    for name, model_func in linear_models.items():
        print(f"{name} (Linear Model) Evaluation:")
        model, y_pred_prob = model_func(X_train, y_train, X_test, y_test)

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)
        model_names.append(name)

        # Save fpr, tpr, and auc for plotting later
        fpr_tpr_auc.append((name, fpr, tpr, auc_score))

        # Print classification report
        print(classification_report(y_test, model.predict(X_test)))

    # Evaluate non-linear models
    for name, model_func in nonlinear_models.items():
        print(f"{name} (Non-Linear Model) Evaluation:")
        model, y_pred_prob = model_func(X_train, y_train, X_test, y_test)

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)
        model_names.append(name)

        # Save fpr, tpr, and auc for plotting later
        fpr_tpr_auc.append((name, fpr, tpr, auc_score))

        # Print classification report
        print(classification_report(y_test, model.predict(X_test)))

    # Cross-validation boxplot (optional)
    # Add this if you want to include cross-validation scores for linear vs non-linear models

    # ROC curve plot (linear vs non-linear models)
    plt.figure(figsize=(12, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line for random classifier

    for name, fpr, tpr, auc_score in fpr_tpr_auc:
        if name in linear_models:
            # Solid line for linear models
            plt.plot(fpr, tpr, label=f"{name} (Linear, AUC = {auc_score:.2f})", linestyle='-')
        else:
            # Dashed line for non-linear models
            plt.plot(fpr, tpr, label=f"{name} (Non-Linear, AUC = {auc_score:.2f})", linestyle='--')

    # Finalize plots
    plt.title('ROC Curve Comparison: Linear vs Non-Linear Models')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.legend(loc="lower right")  # Add legend with model name and AUC for each model
    plt.show()

    # Return the results
    return auc_scores, model_names

def compare_models_with_feature_selection(feature_selection_dict):
   
    # List of models
    models = {
        "Logistic Regression": logistic_regression,
        "Random Forest": random_forest,
        "Gradient Boosting": gradient_boosting,
        "AdaBoost": adaboost,
        "SVC": svc,
        "KNN": knn,
        "Naive Bayes": naive_bayes
    }  

    
     # Store results for plotting
    results = []  # To store model, FS method, fpr, tpr, auc

    # Evaluate models for each model type
    for model_name, model_func in models.items():
        print(f"\nModel: {model_name}")

        for fs_name, (X_fs, y_fs) in feature_selection_dict.items():
            print(f"  Feature Selection: {fs_name}")

            # Split into training and testing sets for the selected features
            X_train, X_test, y_train, y_test = train_test_split(X_fs, y_fs, test_size=0.3, random_state=42)

            # Train and evaluate the model
            model, y_pred_prob = model_func(X_train, y_train, X_test, y_test)

            # Compute ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            auc_score = auc(fpr, tpr)
            results.append((model_name, fs_name, fpr, tpr, auc_score))

            # Print classification report
            print(classification_report(y_test, model.predict(X_test)))

    # Plot ROC curves for each model
    for model_name in models.keys():
        plt.figure(figsize=(10, 8))
        print(f' ---- model {model_name} ----')
        for _ , fs_name, fpr, tpr, auc_score in filter(lambda x: x[0] == model_name, results):
            plt.plot(fpr, tpr, label=f"{fs_name} (AUC = {auc_score:.2f})")
            print(f'ploting  {fs_name}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
        plt.title(f"ROC Curve: {model_name}")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()

    # Return results for further analysis
    return results