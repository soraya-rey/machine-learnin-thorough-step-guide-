import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, kruskal
from scipy import stats
from scipy.stats import shapiro, kstest, expon


def Data_Overview(X):
    print("Data Overview:")
    print(X.info())  # Check data types, missing values
    print("\nStatistical Summary:")
    print(X.describe())  # Summary statistics of numeric features
    return

def unique_data_types(X):
    """
    Prints the unique data types present in the dataset X and their proportions.
    Args:
    - X (DataFrame): Features dataset.
    
    Returns:
    - None
    """
    # Get the data types of each column
    data_types = X.dtypes.value_counts()
    
    # Total number of columns
    total_columns = X.shape[1]
    
    # Calculate the proportions
    proportions = data_types / total_columns
    
    # Print the unique data types and their proportions
    print("Unique Data Types and Their Proportions:")
    for dtype, count in data_types.items():
        proportion = proportions[dtype]
        print(f"{dtype}: {count} columns ({proportion:.2%} of total columns)")



def Univariate_Analysis(X):
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(X[col], kde=True, bins=30)
        plt.title(f'{col} - Distribution')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x=X[col])
        plt.title(f'{col} - Boxplot')
        plt.show()   
    categorical_cols = X.select_dtypes(include=[object]).columns
    for col in categorical_cols:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=X, x=col)
        plt.title(f'{col} - Count Plot')
        plt.show()
    return

def correlation_heatmap(X):
    correlation_matrix = X.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()


def clusters_heatmap(X):
    # Compute the correlation matrix
    correlation_matrix = X.corr()
    # Apply hierarchical clustering to the correlation matrix
    linkage_matrix = linkage(correlation_matrix, method='ward')
    # Reorder the correlation matrix based on hierarchical clustering
    dendro = dendrogram(linkage_matrix, no_plot=True)
    cluster_order = dendro['leaves']
    # Reorder the correlation matrix based on the clustering
    correlation_matrix = correlation_matrix.iloc[cluster_order, cluster_order]
    # Plot the heatmap with clusters
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt='.2f', xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns)
    plt.title('Correlation Heatmap with Clusters')
    plt.show()


def feature_target_correlation_test(X, y, col_name):
    """
    Function to perform statistical tests between a given feature column and target `y`,
    and visualize the relationship using appropriate plots.
    Args:
    - X (DataFrame): Features dataset.
    - y (Series): Target variable (binary).
    - col_name (str): The name of the feature column to test and visualize.
    """
    feature = X[col_name]
    
    # Check if the feature is numeric or categorical
    if feature.dtype in [np.float64, np.int64]:  # Numeric feature
        # Perform t-test for binary target
        group1 = feature[y == 0]  # Class 0
        group2 = feature[y == 1]  # Class 1
        t_stat, p_value = ttest_ind(group1, group2)
        print(f"T-test for {col_name}:")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
        
        # Visualize relationship with scatterplot and boxplot
        plt.figure(figsize=(10, 6))
        # Scatterplot
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=feature, y=y)
        plt.title(f'{col_name} vs Target (Scatterplot)')
        
        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=y, y=feature)
        plt.title(f'{col_name} vs Target (Boxplot)')
        plt.show()
        
    elif feature.dtype == 'object':  # Categorical feature
        # Perform Kruskal-Wallis test for binary target
        group1 = feature[y == 0]  # Class 0
        group2 = feature[y == 1]  # Class 1
        stat, p_value = kruskal(group1, group2)
        print(f"Kruskal-Wallis test for {col_name}:")
        print(f"Statistic: {stat:.4f}, p-value: {p_value:.4f}")
        
        # Visualize relationship with boxplot or countplot
        plt.figure(figsize=(8, 6))
        # Countplot for categorical feature
        sns.countplot(x=feature, hue=y)
        plt.title(f'{col_name} vs Target (Countplot)')
        plt.show()
        
    else:
        print(f"Unsupported feature type for {col_name}. Only numeric and categorical features are supported.")



def distribution_analysis(X):
    """
    Analyzes and visualizes the distribution of each feature in the dataset X.
    Tests for normality and visualizes the distribution for each feature.
    
    Args:
    - X (DataFrame): Features dataset.
    
    Returns:
    - None
    """
    # Create a plot grid for visualizing distributions
    num_features = X.select_dtypes(include=[np.number]).columns
    cat_features = X.select_dtypes(include=[object]).columns
    
    # Visualize numeric feature distributions
    for col in num_features:

        # Normality test for numeric features
        stat, p_value = stats.shapiro(X[col].dropna())
        print(f"Normality Test for {col}:")
        print(f"Shapiro-Wilk test statistic: {stat:.4f}, p-value: {p_value:.4f}")
        ks_statistic, p_value_ks = kstest(X[col].dropna(), 'expon') 
        print(f"ks test statistic: {ks_statistic:.4f}, p-value: {p_value_ks:.4f}")


        plt.figure(figsize=(12, 6))
        
        # Plot Histogram and KDE
        plt.subplot(1, 2, 1)
        sns.histplot(X[col], kde=True, bins=30)
        plt.title(f'{col} - Distribution (Histogram & KDE) \n Shapiro-Wilk test statistic: {stat:.4f}, p-value: {p_value:.4f} \n ks test statistic: {ks_statistic:.4f}, p-value: {p_value_ks:.4f}')
        
        # Plot Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=X[col])
        plt.title(f'{col} - Boxplot')
        
        plt.tight_layout()
        plt.show()       
        
        if p_value < 0.05:
            print(f"{col} is not normally distributed.\n")
        else:
            print(f"{col} appears to be normally distributed.\n")

    # Visualize categorical feature distributions
    for col in cat_features:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=X, x=col)
        plt.title(f'{col} - Count Plot (Categorical Feature)')
        plt.show()
        
        # Frequency of categories
        print(f"Categorical feature '{col}' distribution:")
        print(X[col].value_counts(), "\n")
      
    return

def test_distribution(data):
    """
    Tests for normality and exponential distribution.

    Args:
        data (array-like): The data to be tested.

    Returns:
        str: 
            - 'Normal' if normally distributed, 
            - 'Exponential' if exponentially distributed, 
            - 'Other' otherwise.
    """

    # Test for normality (Shapiro-Wilk)
    stat, p_normal = shapiro(data)
    if p_normal > 0.05:
        return 'Normal'

    # Test for exponential distribution
    shifted_data = data - data.min()  # Ensure non-negative values
    ks_stat, p_exp = kstest(shifted_data, 'expon')
    if p_exp > 0.05:
        return 'Exponential'

    # If neither test passes
    return 'Other'


def distribution_analysis_of_repartition(X):
    """
    Analyzes and visualizes the distribution of each feature in the dataset X.
    Tests for normality, exponential, and other distributions.

    Args:
        X (DataFrame): Features dataset.

    Returns:
        None
    """

    num_features = X.select_dtypes(include=[np.number]).columns
    dist_counts = {'Normal': 0, 'Exponential': 0, 'Other': 0}

    for col in num_features:
        data = X[col].dropna()  # Exclude missing values
        if len(data) > 0:  # Check for non-empty data
            dist = test_distribution(data)
            dist_counts[dist] += 1
        else:
            print(f"Warning: Column '{col}' has no non-null values. Skipping distribution test.")

    # Visualize distribution proportions
    plt.figure(figsize=(8, 8))
    plt.pie(dist_counts.values(), labels=dist_counts.keys(), autopct="%1.1f%%")
    plt.title("Distribution Proportion of Numeric Features")
    plt.show()

    print("Distribution Counts:", dist_counts)


def plot_label_distribution(y):
  """
  Plots the proportion of labels (0 and 1) in a bar chart.

  Args:
    y: A numpy array or list containing the labels (0 and 1).

  Returns:
    None
  """

  # Calculate the proportion of each label
  label_counts = np.bincount(y)
  proportions = label_counts / len(y)

  # Create the bar chart
  plt.figure(figsize=(8, 6))
  plt.bar([0, 1], proportions, color=['blue', 'orange'])
  plt.xlabel("Labels")
  plt.ylabel("Proportion")
  plt.xticks([0, 1], labels=["0", "1"])
  plt.title("Proportion of Labels")
  plt.show()