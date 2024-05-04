"""
Introduction

In this challenge, you'll get the opportunity to tackle one of the most industry-relevant maching learning problems with a unique dataset that will put your modeling skills to the test. Subscription services are leveraged by companies across many industries, from fitness to video streaming to retail. One of the primary objectives of companies with subscription services is to decrease churn and ensure that users are retained as subscribers. In order to do this efficiently and systematically, many companies employ machine learning to predict which users are at the highest risk of churn, so that proper interventions can be effectively deployed to the right audience.

In this challenge, we will be tackling the churn prediction problem on a very unique and interesting group of subscribers on a video streaming service!

Imagine that you are a new data scientist at this video streaming company and you are tasked with building a model that can predict which existing subscribers will continue their subscriptions for another month. We have provided a dataset that is a sample of subscriptions that were initiated in 2021, all snapshotted at a particular date before the subscription was cancelled. Subscription cancellation can happen for a multitude of reasons, including:
* the customer completes all content they were interested in, and no longer need the subscription
* the customer finds themselves to be too busy and cancels their subscription until a later time
* the customer determines that the streaming service is not the best fit for them, so they cancel and look for something better suited

Regardless the reason, this video streaming company has a vested interest in understanding the likelihood of each individual customer to churn in their subscription so that resources can be allocated appropriately to support customers. In this challenge, you will use your machine learning toolkit to do just that!
"""

# %% md
# Customer Churn Prediction with Classical Machine Learning Models
# %% md
## Import Libraries
# %%
# Data packages
import numpy as np
import pandas as pd
import scipy.stats as ss

# Machine Learning / Classification packages
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Visualization Packages
from matplotlib import pyplot as plt
import seaborn as sns
# %% md
## Functions
# %%
def print_dataframe_stats(df: pd.DataFrame) -> None:
    """Print various statistics about a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to print statistics for.
    """

    print(f"Rows   : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")
    print(df.columns)
    print("\nFeatures : \n", df.columns.tolist())
    print("\nUnique values : \n", df.nunique())
    print("\nMissing values Total : ", df.isnull().sum().sum())
    print("\nMissing values : \n", df.isnull().sum())
    print("\nType of values: \n", df.dtypes)


# %%
def calculate_average_churn_in_bins(df, column_name, bin_size=10, target_column='Churn'):
    """
    Calculate the average churn rate within specified bins for a given DataFrame column.

    Args:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The column to create bins from.
    target_column (str): The column to calculate average churn rate. Defaults to 'Churn'.
    bin_size (int): The width of each bin. Defaults to 10.

    Returns:
    tuple: A tuple containing two lists -
           1. List of average churn rates for non-empty bins.
           2. List of the start values of each non-empty bin.
    """
    # Define the bins based on the specified column and step size
    bins = np.arange(df[column_name].min(), df[column_name].max() + bin_size, bin_size)

    # Assign bin indices for each data point
    bin_indices = np.digitize(df[column_name], bins)

    # Initialize lists for storing results
    average_churn_rates = []
    valid_bins = []

    # Iterate over each bin and calculate the average churn rate for non-empty bins
    for bin_index in range(1, len(bins)):
        # Get data points in the current bin
        data_in_bin = df[bin_indices == bin_index]

        # Proceed if the bin has data points
        if not data_in_bin.empty:
            avg_churn = data_in_bin[target_column].mean()  # Calculate average churn rate
            average_churn_rates.append(avg_churn)
            valid_bins.append(bins[bin_index - 1])  # Store the start value of the bin

    return average_churn_rates, valid_bins


# %%
def create_bar_plot(dataframe, column_name, xlabel_text, bin_size=10):
    """
    Create a bar plot showing the average churn rate by specified bins.

    Args:
    dataframe (pd.DataFrame): The input DataFrame containing the data.
    column_name (str): The name of the column to create bins from.
    xlabel_text (str): The label for the x-axis.
    bin_size (int): The width of each bin. Defaults to 10.

    Returns:
    None. Displays the bar plot.
    """
    # Calculate average churn rates and bin values
    average_churn_rate, bins = calculate_average_churn_in_bins(dataframe, column_name, bin_size)

    # Format bin labels for display on the x-axis
    formatted_bins = [f'{bin_value:.1f}' for bin_value in bins]

    # Create the bar plot with specified x and y data
    ax = sns.barplot(x=formatted_bins, y=average_churn_rate, color='skyblue')

    # Set the plot labels and title
    ax.set_xlabel(xlabel_text)
    ax.set_ylabel('Average Churn')
    ax.set_title(f'Average Customer Churn by {xlabel_text}')

    # Enable grid for better visibility
    ax.grid(True)

    # Display the plot
    plt.show()


# %%
def get_categorical_columns(df, exclude=None):
    """
    Get list of categorical columns in a DataFrame, excluding specified columns.

    Parameters:
        df (DataFrame): Input DataFrame.
        exclude (list): List of columns to exclude. Default is None.

    Returns:
        list: List of categorical columns.
    """
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if exclude:
        categorical_columns = [col for col in categorical_columns if col not in exclude]
    return categorical_columns


# %%
def plot_categorical_churn_counts(df, categorical_columns, target_column='Churn'):
    """
    Plot count of each category in categorical columns with respect to churn.

    Parameters:
        df (DataFrame): Input DataFrame.
        categorical_columns (list): List of categorical columns to plot.
        target_column (str): Target column to analyze churn. Default is 'Churn'.
    """
    # Set up subplot grid
    num_cols = len(categorical_columns)
    num_rows = num_cols // 2 + num_cols % 2  # Adjust number of rows based on number of columns

    # Create subplots
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))

    # Flatten axes array
    axes = axes.flatten()

    # Iterate over categorical columns and create bar charts
    for i, column in enumerate(categorical_columns):
        ax = axes[i]
        sns.countplot(x=column, hue=target_column, data=df, ax=ax)
        ax.set_title(f'Count of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Count')

    # If there are unused subplots, remove them
    for i in range(num_cols, num_rows * 2):
        fig.delaxes(axes[i])

    # Adjust layout
    plt.tight_layout()
    plt.show()


# %%
def plot_heatmap(df):
    """
    Plot heatmap to visualize correlations between numeric columns.

    Parameters:
        df (DataFrame): Input DataFrame.
        numeric_columns (list): List of numeric columns to analyze.
    """
    # Calculate correlation matrix for numeric columns
    numeric_columns = df.select_dtypes(include=['int', 'float']).columns
    corr_matrix = df[numeric_columns].corr()

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(corr_matrix, annot=True, cmap='PiYG', fmt=".2f", annot_kws={"size": 12})
    plt.title('Correlation between Numeric Columns')

    # Rotate x-axis labels to 45 degrees
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.show()


# %%
def categorical_correlation(df, cat_cols):
    """
    Calculate the Cramer's V statistic for each categorical column in relation to the 'Churn' column.

    Parameters:
        df (DataFrame): Input DataFrame.
        cat_cols (list): List of categorical column names.

    Returns:
        dict: Dictionary containing Cramer's V statistic for each categorical column.
    """

    def cramers_corrected_stat(confusion_matrix):
        """
        Calculate Cramer's V statistic for categorical-categorical association.
        Uses correction from Bergsma and Wicher.

        Parameters:
            confusion_matrix (array): Contingency table (cross-tabulation).

        Returns:
            float: Cramer's V statistic.
        """
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    scores = {}
    for col in cat_cols:
        crosstab = pd.crosstab(df[col], df['Churn']).values
        scores[col] = cramers_corrected_stat(crosstab)

    return scores


# %%
def calculate_churn_rate(df, categorical_column, target_column='Churn'):
    """
    Calculate churn rate for each unique value in a categorical column.

    Parameters:
        df (DataFrame): Input DataFrame.
        categorical_column (str): Name of the categorical column.
        target_column (str): Name of the target column. Default is 'Churn'.

    Returns:
        DataFrame: DataFrame containing unique values of the categorical column and their corresponding churn rates.
    """
    # Group data by the categorical column and calculate churn rate
    churn_rates = df.groupby(categorical_column)[target_column].mean().reset_index()
    churn_rates.rename(columns={target_column: 'Churn Rate'}, inplace=True)

    return churn_rates


# %%
def check_column_exists(df, column_name):
    """
    Validates whether a specified column exists in a DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame to check.
    column_name (str): The name of the column to validate.

    Raises:
    ValueError: If the specified column does not exist in the DataFrame.
    """
    if column_name not in df.columns:
        raise ValueError(f"The column '{column_name}' does not exist in the DataFrame.")


# %%
def calculate_churn_rate_by_bins(df, numeric_column, num_bins=10, target_column='Churn'):
    """
    Calculate the average churn rate in specified bins for a given DataFrame.

    Args:
    df (pd.DataFrame): The input DataFrame.
    numeric_column (str): The name of the numeric column to create bins from.
    num_bins (int): The number of bins to create. Default is 10.
    target_column (str): The column to calculate churn rate from. Default is 'Churn'.

    Returns:
    pd.DataFrame: A DataFrame with bins and the corresponding churn rates.

    Raises:
    ValueError: If the target column does not exist in the DataFrame.
    """
    # Ensure the target column exists
    check_column_exists(df, target_column)

    # Drop NaN values for the numeric column and create bins
    df_clean = df.dropna(subset=[numeric_column])
    bins = pd.cut(df_clean[numeric_column], bins=num_bins, include_lowest=True)

    # Group by bins and calculate the average churn rate
    churn_rates = df_clean.groupby(bins, observed=False)[target_column].mean().reset_index()

    # Check for column name conflict and rename if necessary
    churn_rate_column = 'Churn Rate'
    if churn_rate_column in churn_rates.columns:
        churn_rate_column = f"{churn_rate_column} (Unique)"

    churn_rates.rename(columns={target_column: churn_rate_column}, inplace=True)

    return churn_rates


# %%
def apply_one_hot_encoding(df, target_column):
    """
    Create a one-hot encoded version of a specified column in a DataFrame.

    This function takes a DataFrame and a target column name as input,
    then performs one-hot encoding on the target column. The one-hot encoded
    columns are concatenated with the original DataFrame, and the original
    target column is dropped. Additionally, the resulting DataFrame is
    converted to numeric values to ensure consistent data types.

    Args:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the column to be one-hot encoded.

    Returns:
    pd.DataFrame: A DataFrame with the one-hot encoded columns added and the
                  original target column removed.
    """
    # Perform one-hot encoding
    one_hot_encoded = pd.get_dummies(df[target_column])

    # Concatenate the one-hot encoded columns with the original DataFrame
    df_encoded = pd.concat([df, one_hot_encoded], axis=1)
    df_encoded = df_encoded.drop(columns=target_column)
    df_encoded = df_encoded * 1

    return df_encoded


# %%
def extract_features(dataframe, columns_to_exclude):
    """
    Extract features by removing specified columns from the DataFrame.

    Args:
    dataframe (pd.DataFrame): The source DataFrame from which features are extracted.
    columns_to_exclude (list of str): Columns to be excluded (usually target variables or identifiers).

    Returns:
    pd.DataFrame: A DataFrame containing only the features (with specified columns removed).
    """
    return dataframe.drop(columns=columns_to_exclude)


# %%
def extract_target(dataframe, target_column):
    """
    Extract the target column from a DataFrame.

    Args:
    dataframe (pd.DataFrame): The source DataFrame.
    target_column (str): The name of the target column to extract.

    Returns:
    pd.Series: A Series containing the target variable.
    """
    return dataframe[target_column]


# %%
# Function to define a parameter grid for Random Forest
def get_rf_param_grid():
    """
    Define the parameter grid for Random Forest GridSearchCV.

    Returns:
    dict: A dictionary with parameter options for Random Forest.
    """
    return {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'n_jobs': [-1]
    }


# %%
# Function to perform GridSearchCV to get the best Random Forest model
def get_best_random_forest(x_train, y_train, param_grid):
    """
    Perform grid search to find the best Random Forest model.

    Args:
    x_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    param_grid (dict): Parameter grid for grid search.

    Returns:
    tuple: Best parameters and the best Random Forest model.
    """
    rf_model = RandomForestClassifier()
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    return grid_search.best_params_, grid_search.best_estimator_


# %%
# Function to define comparison models
def get_comparison_models(best_rf_model):
    """
    Define a set of models for comparison, including the best Random Forest model.

    Args:
    best_rf_model (RandomForestClassifier): The best Random Forest model.

    Returns:
    dict: A dictionary containing the models for comparison.
    """
    return {
        'Linear Regression': LinearRegression(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'Support Vector Machine': SVR(),
        'Random Forest': best_rf_model
    }


# %%
# Function to evaluate models and return the best model
def find_best_model(models, x_train, y_train, x_test, y_test):
    """
    Evaluate models and return the one with the best performance.

    Args:
    models (dict): A dictionary of models to evaluate.
    x_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    x_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test target.

    Returns:
    tuple: The best model and its name.
    """
    best_model_name = None
    best_score = float('inf')

    for name, model in models.items():
        model.fit(x_train, y_train)

        # Predict and evaluate on the test set
        y_pred = model.predict(x_test)
        score = evaluate_model_performance(y_test, y_pred, x_train, y_train)

        print(f"{name}: Performance = {score}")

        # If this model's score is better, update best score and best model name
        if score < best_score:
            best_score = score
            best_model_name = name

    return models[best_model_name], best_model_name


# %%
def evaluate_model_performance(model, x_train, y_train, y_test, cv=5, scoring='neg_mean_squared_error'):
    """
    Evaluate the performance of a machine learning model on test data.

    This function calculates various evaluation metrics such as accuracy,
    precision, recall, F1 score, confusion matrix, and cross-validation scores.

    Args:
    model: The machine learning model to evaluate.
    x_train: The training feature set.
    y_train: The training labels.
    x_test: The test feature set.
    y_test: The test labels.
    cv (int): The number of cross-validation folds. Default is 5.
    scoring (str): The scoring metric for cross-validation. Default is 'neg_mean_squared_error'.

    Returns:
    float: The mean of the cross-validation scores.
    """
    # Calculate basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calculate cross-validation scores
    cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring=scoring)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)
    print(f"Mean Squared Error (CV) = {cv_scores.mean()}")

    # Return the mean cross-validation score
    return cv_scores.mean()


# %%
def plot_feature_importance(model, feature_names, model_name, figsize=(15, 10), font_scale=1.5, fontsize=10):
    """
    Plot the feature importance for a given model.

    This function generates a bar plot of feature importance using Seaborn.
    It takes a model, a list of feature names, and a model name to create a
    visualization of the feature importance.

    Args:
    model: The trained model with a `feature_importance_` attribute.
    feature_names (list of str): The list of feature names.
    model_name (str): The name of the model (used in the plot title).
    figsize (tuple): The size of the plot. Default is (20, 16).
    font_scale (float): The scale factor for plot fonts. Default is 1.5.

    Returns:
    None. Displays the plot.
    """

    # Create a DataFrame with feature importance
    feature_importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_,
    })

    # Increase font size for all elements in the plot
    sns.set(font_scale=font_scale)

    # Plot the feature importance
    plt.figure(figsize=figsize)
    feature_importances_df.sort_values(by='Importance', ascending=False, inplace=True)
    sns.barplot(x='Importance', y='Feature', data=feature_importances_df, palette='viridis')

    # Set plot titles and labels
    plt.title(f"Feature Importance ({model_name})", fontsize=15)
    plt.xlabel('Importance', fontsize)
    plt.ylabel('Feature', fontsize)
    plt.xticks(fontsize)
    plt.yticks(fontsize)

    # Display the plot
    plt.show()


# %% md
## Import and Read Files
# %%
df_prediction_submission = pd.read_csv('Files/prediction_submission.csv')
# %%
data_descriptions = pd.read_csv('Files/data_descriptions.csv')
pd.set_option('display.max_colwidth', None)
data_descriptions
# %%
train_df = pd.read_csv("Files/train.csv")
print('train_df Shape:', train_df.shape)
train_df.head()
# %%
test_df = pd.read_csv("Files/test.csv")
print('test_df Shape:', test_df.shape)
test_df.head()
# %% md
## EDA
# %%
print_dataframe_stats(train_df)
# %%
print_dataframe_stats(test_df)
# %%
train_df.describe()
# %% md
## Visualize Features
# %%
create_bar_plot(train_df, 'AccountAge', 'Account Age')
# %%
create_bar_plot(train_df, 'MonthlyCharges', 'Monthly Charges', 2)
# %%
create_bar_plot(train_df, 'TotalCharges', 'Total Charges', 20)
# %%
create_bar_plot(train_df, 'ViewingHoursPerWeek', 'Viewing Hours Per Week', 4)
# %%
create_bar_plot(train_df, 'AverageViewingDuration', 'Average Viewing Duration', 15)
# %%
create_bar_plot(train_df, 'UserRating', 'User Rating', 1)
# %%
create_bar_plot(train_df, 'SupportTicketsPerMonth', 'Support Tickets Per Month', 1)
# %%
create_bar_plot(train_df, 'ContentDownloadsPerMonth', 'Content Downloads Per Month', 5)
# %%
# Get list of categorical columns excluding 'CustomerID'
categorical_columns = get_categorical_columns(train_df, exclude=['CustomerID'])
print(categorical_columns)
# %%
plot_categorical_churn_counts(train_df, categorical_columns)
# %%
plot_heatmap(train_df)
# %% md
## Calculate Correlation
# %%
correlation_scores = categorical_correlation(train_df, categorical_columns)
print(correlation_scores)
# %% md
## Calculate Churn Rate
# %%
for categorical_column in categorical_columns:
    churn_rates = calculate_churn_rate(train_df, categorical_column)
    print(churn_rates)
# %%
try:
    numeric_columns = train_df.select_dtypes(include=['int', 'float']).columns

    for numeric_column in numeric_columns:
        churn_rates_by_bins = calculate_churn_rate_by_bins(train_df, numeric_column)
        print(churn_rates_by_bins)

except Exception as ex:
    print(f"An error occurred while processing the numeric columns: {ex}")
# %% md
## Encode DataFrames
# %%
df_encoded_train = apply_one_hot_encoding(train_df, categorical_columns)
df_encoded_train
# %%
df_encoded_test = apply_one_hot_encoding(test_df, categorical_columns)
df_encoded_test
# %% md
## Split Train - Test
# %%
# Define the columns to exclude for feature extraction
columns_to_exclude_train = ['Churn', 'CustomerID']
columns_to_exclude_test = ['CustomerID']

# Extract features and target for training data
x_train = extract_features(df_encoded_train, columns_to_exclude_train)
y_train = extract_target(df_encoded_train, 'Churn')

# Extract features for test data
x_test = extract_features(df_encoded_test, columns_to_exclude_test)

# Extract target for test/prediction data (assuming it's used for inference or submission)
y_test = extract_features(df_prediction_submission, columns_to_exclude_test)
# %% md
## Evaluate Models
# %%
# Define the parameter grid and get the best Random Forest model
param_grid = get_rf_param_grid()
best_params, best_rf_model = get_best_random_forest(x_train, y_train, param_grid)
# %%
# Get the comparison models, including the best Random Forest
comparison_models = get_comparison_models(best_rf_model)

# Find the best model among the comparison models
best_model, best_model_name = find_best_model(comparison_models, x_train, y_train, x_test, y_test)

# Display information about the best model
print(f"The best model is: {best_model_name} with the best performance.")
# %%
# Fit and predict with the best model
best_model.fit(x_train, y_train)
y_test_pred = best_model.predict(x_test)

# Plot feature importance if applicable
if hasattr(best_model, 'feature_importance_'):
    plot_feature_importance(best_model, x_train.columns.tolist(), best_model_name)
# %%

# %%
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'n_jobs': [-1]
}
# %%
model = RandomForestClassifier()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(x_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train the model with the best parameters
best_model = RandomForestClassifier(**best_params)
best_model.fit(x_train, y_train)

# Predict and evaluate
y_pred = best_model.predict(x_test)
# %%
evaluate_model_performance(y_test, y_pred, x_train, y_train)
# %%
plot_feature_importance(best_model, x_train, 'Random Forest')
# %%
# Initialize variables to keep track of the best model and its score
best_model_name = None
best_score = float('inf')
print(best_score)
# %%
# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Support Vector Machine': SVR(),
    'RandomForest': RandomForestClassifier()
    # 'Neural Network': MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

# Evaluate each model
for name, model in models.items():
    model.fit(x_train, y_train)
    score = evaluate_model_performance(y_test, y_pred, x_train, y_train)
    print(f"{name}: Mean Squared Error = {score}")

    if score < best_score:
        best_score = score
        best_model_name = name
# %%
best_model = models[best_model_name]
# %%
# Now, best_model contains the best model based on the lowest mean squared error
print(f"The best model is: {best_model_name} with a Mean Squared Error of {best_score}")

# Fit the best model to the training data and make predictions
best_model.fit(x_train, y_train)
y_test_pred = best_model.predict(x_test)
# %%
results = permutation_importance(model, x_train, y_train, scoring='neg_mean_squared_error')
# %%
importance = results.importances_mean
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))
# %%
plt.bar([x for x in range(len(importance))], importance)
plt.show()