
# Import Libraries
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from matplotlib import pyplot as plt


# Utility Functions
class UtilityFunctions:
    @staticmethod
    def print_dataframe_stats(df: pd.DataFrame) -> None:
        """Print basic statistics about a DataFrame."""
        print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        print("Features:", df.columns.tolist())
        print("Unique Values:", df.nunique())
        print("Total Missing Values:", df.isnull().sum().sum())
        print("Missing Values by Column:", df.isnull().sum())
        print("Data Types:", df.dtypes)

    @staticmethod
    def plot_heatmap(df):
        """Plot a heatmap to visualize correlations among numeric columns."""
        # Calculate correlation matrix for numeric columns
        numeric_columns = df.select_dtypes(include=['int', 'float']).columns
        corr_matrix = df[numeric_columns].corr()

        # Plot heatmap
        plt.figure(figsize=(20, 12))
        ax = sns.heatmap(corr_matrix, annot=True, cmap='PiYG', fmt=".2f", annot_kws={"size": 12})
        plt.title('Correlation between Numeric Columns')

        # Rotate x-axis labels to 45 degrees
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.show()


class CalculateChurnRate:
    def check_column_exists(self, df, column_name):
        """Validate whether a specified column exists in a DataFrame."""
        if column_name not in df.columns:
            raise ValueError(f"The column '{column_name}' does not exist in the DataFrame.")

    def calculate_average_churn_in_bins(self, df, column_name, bin_size=10, target_column='Churn'):
        """Calculate the average churn rate within specified bins for a given DataFrame column."""
        bins = np.arange(df[column_name].min(), df[column_name].max() + bin_size, bin_size)
        bin_indices = np.digitize(df[column_name], bins)

        average_churn_rates = []
        valid_bins = []

        for bin_index in range(1, len(bins)):
            data_in_bin = df[bin_indices == bin_index]
            if not data_in_bin.empty:
                avg_churn = data_in_bin[target_column].mean()
                average_churn_rates.append(avg_churn)
                valid_bins.append(bins[bin_index - 1])

        return average_churn_rates, valid_bins

    def create_bar_plot(self, df, column_name, xlabel_text, bin_size=10):
        """Create a bar plot showing the average churn rate by specified bins."""
        average_churn_rate, bins = self.calculate_average_churn_in_bins(df, column_name, bin_size)
        formatted_bins = [f'{bin:.1f}' for bin in bins]

        sns.barplot(x=formatted_bins, y=average_churn_rate, color='skyblue')
        plt.xlabel(xlabel_text)
        plt.ylabel('Average Churn')
        plt.title(f'Average Customer Churn by {xlabel_text}')
        plt.grid(True)
        plt.show()

    def get_categorical_columns(self, df, exclude=None):
        """Get a list of categorical columns in a DataFrame, excluding specified ones."""
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        if exclude:
            return [col for col in categorical_columns if col not in exclude]
        return categorical_columns

    def plot_categorical_churn_counts(self, df, categorical_columns, target_column='Churn'):
        """Plot the count of each category in categorical columns with respect to churn."""
        num_cols = len(categorical_columns)
        num_rows = (num_cols + 1) // 2

        fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
        axes = axes.flatten()

        for i, column in enumerate(categorical_columns):
            sns.countplot(x=column, hue=target_column, data=df, ax=axes[i])
            axes[i].set_title(f'Count of {column}')

        if len(axes) > num_cols:
            fig.delaxes(axes[-1])

        plt.tight_layout()
        plt.show()

    def cramers_corrected_stat(self, confusion_matrix):
        """Calculate Cramer's V statistic with correction."""
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min(kcorr - 1, rcorr - 1))

    def categorical_correlation(self, df, cat_cols):
        """Calculate Cramer's V for each categorical column relative to the 'Churn' column."""
        scores = {}
        for col in cat_cols:
            crosstab = pd.crosstab(df[col], df['Churn']).values
            scores[col] = self.cramers_corrected_stat(crosstab)
        return scores

    def calculate_churn_rate(self, df, categorical_column, target_column='Churn'):
        """Calculate churn rate for each unique value in a categorical column."""
        churn_rates = df.groupby(categorical_column)[target_column].mean().reset_index()
        churn_rates.rename(columns={target_column: 'Churn Rate'}, inplace=True)
        return churn_rates

    def calculate_churn_rate_by_bins(self, df, numeric_column, num_bins=10, target_column='Churn'):
        """Calculate the average churn rate by specified bins for a given numeric column."""
        self.check_column_exists(df, target_column)

        df_clean = df.dropna(subset=[numeric_column])
        bins = pd.cut(df_clean[numeric_column], bins=num_bins, include_lowest=True)

        churn_rates = df_clean.groupby(bins, observed=False)[target_column].mean().reset_index()
        churn_rates.rename(columns={target_column: 'Churn Rate'}, inplace=True)

        return churn_rates


# Data Processing Functions
class DataProcessingFunctions:
    @staticmethod
    def apply_one_hot_encoding(df, target_column):
        """Apply one-hot encoding to a specified column in a DataFrame."""
        # Perform one-hot encoding
        one_hot_encoded = pd.get_dummies(df[target_column])

        # Concatenate the one-hot encoded columns with the original DataFrame
        df_encoded = pd.concat([df, one_hot_encoded], axis=1)
        df_encoded = df_encoded.drop(columns=target_column)
        df_encoded = df_encoded * 1

        return df_encoded

    @staticmethod
    def extract_features(df, columns_to_exclude):
        """Extract features by removing specified columns from a DataFrame."""
        return df.drop(columns=columns_to_exclude)

    @staticmethod
    def extract_target(df, target_column):
        """Extract the target variable from a DataFrame."""
        return df[target_column]


# Model Training and Evaluation Functions
class ModelTrainingEvaluationFunctions:
    # Constants
    def __init__(self):
        self.DEFAULT_CV = 5  # Default cross-validation folds
        self.DEFAULT_SCORING = 'f1'  # Default scoring metric

    def evaluate_classification_performance(self, model, x_train, y_train, y_test, y_pred, cv=None):
        """Evaluate the performance of a classifier model using accuracy and F1-score."""

        if cv is None:
            cv = self.DEFAULT_CV

        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=1)
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')

        # Cross-validation accuracy for model stability
        cv_accuracy = cross_val_score(model, x_train, y_train, cv=cv, scoring=self.DEFAULT_SCORING).mean()

        # Display metrics
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Cross-Validation Accuracy:", cv_accuracy)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("AUC-ROC:", roc_auc_score(y_test, y_pred))

        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "cross_val_accuracy": cv_accuracy,
        }

    def get_best_model_and_params(self, x_train, y_train, x_test, y_test):
        """Find the best model among comparison models using GridSearchCV."""
        models = {
            'Random Forest': (RandomForestClassifier(), {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                # 'n_jobs': [-1],
            }),
            'Gradient Boosting Classifier': (GradientBoostingClassifier(), {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }),
            'Support Vector Classifier': (SVC(), {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto'],
            }),
            'Logistic Regression': (LogisticRegression(), {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear'],
            }),
            'K-Nearest Neighbors': (KNeighborsClassifier(), {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            })
        }

        best_model = None
        best_params = None
        best_score = -1
        best_evaluation_results = {}

        for name, (model, param_grid) in models.items():
            grid_search = GridSearchCV(model, param_grid, cv=5, verbose=1, n_jobs=-1)
            grid_search.fit(x_train, y_train)

            # Evaluate the model
            evaluation_results = self.evaluate_classification_performance(grid_search.best_estimator_, x_train, y_train, x_test, y_test)

            if evaluation_results['f1_score'] > best_score:
                best_score = evaluation_results['f1_score']
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_evaluation_results = evaluation_results

        return best_model, best_params, best_evaluation_results

    def plot_feature_importance(self, model, feature_names, model_name, figsize=(15, 10), font_scale=1.5, fontsize=10):
        """Plot the feature importance for a given model."""
        # Check the type of model
        if isinstance(model, RandomForestClassifier) or isinstance(model, GradientBoostingClassifier):
            # For RandomForestClassifier or GradientBoostingClassifier
            feature_importances_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_,
            })
        elif isinstance(model, KNeighborsClassifier):
            # For KNeighborsClassifier
            # Compute permutation importance
            result = permutation_importance(model, x_test, y_test, n_repeats=10, random_state=42)
            feature_importances = result.importances_mean
            feature_importances_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances,
            })
        else:
            # For other models (assuming they don't have a direct feature importance attribute)
            print("Warning: Feature importance not available for this model.")
            return

        # Increase font size for all elements in the plot
        sns.set(font_scale=font_scale)

        # Plot the feature importance
        plt.figure(figsize=figsize)
        feature_importances_df.sort_values(by='Importance', ascending=False, inplace=True)
        sns.barplot(x='Importance', y='Feature', data=feature_importances_df, legend=False)

        # Set plot titles and labels
        plt.title(f"Feature Importance ({model_name})", fontsize=15)
        plt.xlabel('Importance', fontsize=fontsize)
        plt.ylabel('Feature', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # Display the plot
        plt.show()


df_prediction_submission = pd.read_csv('Files/prediction_submission.csv')
data_descriptions = pd.read_csv('Files/data_descriptions.csv')
pd.set_option('display.max_colwidth', None)

train_df = pd.read_csv("Files/train.csv")
print('train_df Shape:', train_df.shape)

test_df = pd.read_csv("Files/test.csv")
print('test_df Shape:', test_df.shape)

## EDA
UtilityFunctions().print_dataframe_stats(train_df)
UtilityFunctions().print_dataframe_stats(test_df)
 
## Visualize Features
CalculateChurnRate().create_bar_plot(train_df, 'AccountAge', 'Account Age')
CalculateChurnRate().create_bar_plot(train_df, 'MonthlyCharges', 'Monthly Charges', 2)
CalculateChurnRate().create_bar_plot(train_df, 'TotalCharges', 'Total Charges', 20)
CalculateChurnRate().create_bar_plot(train_df, 'ViewingHoursPerWeek', 'Viewing Hours Per Week', 4)
CalculateChurnRate().create_bar_plot(train_df, 'AverageViewingDuration', 'Average Viewing Duration', 15)
CalculateChurnRate().create_bar_plot(train_df, 'UserRating', 'User Rating', 1)
CalculateChurnRate().create_bar_plot(train_df, 'SupportTicketsPerMonth', 'Support Tickets Per Month', 1)
CalculateChurnRate().create_bar_plot(train_df, 'ContentDownloadsPerMonth', 'Content Downloads Per Month', 5)

# Get list of categorical columns excluding 'CustomerID'
categorical_columns = CalculateChurnRate().get_categorical_columns(train_df, exclude=['CustomerID'])
print(categorical_columns)

CalculateChurnRate().plot_categorical_churn_counts(train_df, categorical_columns)

UtilityFunctions.plot_heatmap(train_df)
 
## Calculate Correlation
correlation_scores = CalculateChurnRate().categorical_correlation(train_df, categorical_columns)
print(correlation_scores)

## Calculate Churn Rate
for categorical_column in categorical_columns:
    churn_rates = CalculateChurnRate().calculate_churn_rate(train_df, categorical_column)
    print(churn_rates)

try:
    numeric_columns = train_df.select_dtypes(include=['int', 'float']).columns

    for numeric_column in numeric_columns:
        churn_rates_by_bins = CalculateChurnRate().calculate_churn_rate_by_bins(train_df, numeric_column)
        print(churn_rates_by_bins)

except Exception as ex:
    print(f"An error occurred while processing the numeric columns: {ex}")
 
## Encode DataFrames
df_encoded_train = DataProcessingFunctions().apply_one_hot_encoding(train_df, categorical_columns)
df_encoded_test = DataProcessingFunctions().apply_one_hot_encoding(test_df, categorical_columns)

## Split Train - Test
# Define the columns to exclude for feature extraction
columns_to_exclude_train = ['Churn', 'CustomerID']
columns_to_exclude_test = ['CustomerID']

# Extract features and target for training data
x_train = DataProcessingFunctions().extract_features(df_encoded_train, columns_to_exclude_train)
y_train = DataProcessingFunctions().extract_target(df_encoded_train, 'Churn')

# Extract features for test data
x_test = DataProcessingFunctions().extract_features(df_encoded_test, columns_to_exclude_test)

# Extract target for test/prediction data (assuming it's used for inference or submission)
y_test = DataProcessingFunctions().extract_features(df_prediction_submission, columns_to_exclude_test)
 
## Evaluate Models
# Find the best model among the comparison models
best_model, best_params, best_evaluation_results = ModelTrainingEvaluationFunctions().get_best_model_and_params(x_train, y_train, x_test, y_test)

# Display information about the best model
print(f"The best model is: {best_model} with the best performance.")

# Fit and predict with the best model
best_model.fit(x_train, y_train)
y_test_pred = best_model.predict(x_test)

ModelTrainingEvaluationFunctions().plot_feature_importance(best_model, x_train.columns.tolist(), best_model)
