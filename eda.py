import numpy as np
import pandas as pd
import scipy.stats as ss
import os
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


class ChurnAnalysis:
    def __init__(self, data_path, target_column='Churn', graphs_dir="graphs_eda", results_dir="results"):
        self.data_path = data_path
        self.target_column = target_column
        self.df =data_path

        self.graphs_dir = graphs_dir
        self.results_dir = results_dir
        os.makedirs(self.graphs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def print_dataframe_stats(self):
        """Print various statistics about the dataset."""
        print(f"Rows   : {self.df.shape[0]}")
        print(f"Columns : {self.df.shape[1]}")
        print("\nFeatures : \n", self.df.columns.tolist())
        print("\nUnique values : \n", self.df.nunique())
        print("\nMissing values Total : ", self.df.isnull().sum().sum())
        print("\nMissing values : \n", self.df.isnull().sum())
        print("\nType of values: \n", self.df.dtypes)

    def calculate_average_churn_in_bins(self, column_name, bin_size=10):
        """
        Calculate average churn rate in bins for a numerical column.
        """
        bins = np.arange(self.df[column_name].min(), self.df[column_name].max() + bin_size, bin_size)
        bin_indices = np.digitize(self.df[column_name], bins)

        average_churn_rates, valid_bins = [], []

        for bin_index in range(1, len(bins)):
            data_in_bin = self.df[bin_indices == bin_index]
            if not data_in_bin.empty:
                avg_churn = data_in_bin[self.target_column].mean()
                average_churn_rates.append(avg_churn)
                valid_bins.append(bins[bin_index - 1])

        return average_churn_rates, valid_bins

    def create_bar_plot(self, column_name, xlabel_text, bin_size=10):
        """
        Create and save a bar plot showing average churn rate by specified bins.
        """
        average_churn_rate, bins = self.calculate_average_churn_in_bins(column_name, bin_size)
        formatted_bins = [f'{bin_value:.1f}' for bin_value in bins]

        plt.figure(figsize=(8, 5))
        ax = sns.barplot(x=formatted_bins, y=average_churn_rate, color='skyblue')
        ax.set_xlabel(xlabel_text)
        ax.set_ylabel('Average Churn')
        ax.set_title(f'Average Customer Churn by {xlabel_text}')
        ax.grid(True)

        plot_path = os.path.join(self.graphs_dir, f"{column_name}_churn_barplot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved: {plot_path}")

    def get_categorical_columns(self, exclude=None):
        """Return categorical columns, excluding specified ones."""
        categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        if exclude:
            categorical_columns = [col for col in categorical_columns if col not in exclude]
        return categorical_columns

    def plot_categorical_churn_counts(self):
        """
        Plot and save count plots for each categorical column with hue to show the values within each column.
        """
        categorical_columns = self.get_categorical_columns(exclude=['CustomerID'])
        num_cols = len(categorical_columns)
        num_rows = (num_cols // 2) + (num_cols % 2)

        fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
        axes = axes.flatten()

        for i, column in enumerate(categorical_columns):
            ax = axes[i]
            sns.countplot(x=column, data=self.df, hue=column, ax=ax, palette="viridis")  # Hue added here
            ax.set_title(f'Count of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Count')
            ax.legend(title=column, loc='upper right', fontsize=8)  # Legend to show categories

        for i in range(num_cols, num_rows * 2):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plot_path = os.path.join(self.graphs_dir, "categorical_churn_counts.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved: {plot_path}")

    def plot_heatmap(self):
        """
        Plot and save heatmap for numeric feature correlations.
        """
        numeric_columns = self.df.select_dtypes(include=['int', 'float']).columns
        corr_matrix = self.df[numeric_columns].corr()

        plt.figure(figsize=(45, 30))
        ax = sns.heatmap(corr_matrix, annot=True, cmap='PiYG', fmt=".2f", annot_kws={"size": 12})
        plt.title('Correlation between Numeric Columns')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plot_path = os.path.join(self.graphs_dir, "heatmap.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved: {plot_path}")

    def categorical_correlation(self):
        """
        Calculate Cramer's V correlation for categorical columns.
        """
        def cramers_corrected_stat(confusion_matrix):
            chi2 = ss.chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)
            return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

        scores = {}
        categorical_columns = self.get_categorical_columns(exclude=['CustomerID'])
        for col in categorical_columns:
            crosstab = pd.crosstab(self.df[col], self.df[self.target_column]).values
            scores[col] = cramers_corrected_stat(crosstab)

        result_path = os.path.join(self.results_dir, "categorical_correlation.json")
        pd.DataFrame.from_dict(scores, orient='index', columns=['CramersV']).to_csv(result_path)
        print(f"Saved: {result_path}")
        return scores

    def calculate_churn_rate(self):
        """
        Calculate churn rate for each category in categorical columns.
        """
        churn_rates = {}
        categorical_columns = self.get_categorical_columns(exclude=['CustomerID'])
        for categorical_column in categorical_columns:
            churn_rates[categorical_column] = self.df.groupby(categorical_column)[self.target_column].mean().reset_index()

        result_path = os.path.join(self.results_dir, "churn_rates.json")
        pd.concat(churn_rates.values()).to_csv(result_path, index=False)
        print(f"Saved: {result_path}")
        return churn_rates

    def run_analysis(self):
        """Execute all analysis steps and save results."""
        self.print_dataframe_stats()
        for col in self.df.select_dtypes(include=['int', 'float']).columns:
            self.create_bar_plot(col, col)
        self.plot_categorical_churn_counts()
        self.plot_heatmap()
        self.categorical_correlation()
        self.calculate_churn_rate()


def handle_categorical_values(df):

    le = LabelEncoder()

    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    for col in object_cols:
        unique_vals = set(df[col].unique())
        if unique_vals.issubset({'yes', 'no'}):
            df[col] = df[col].map({'yes': 1, 'no': 0})

    list_drop = ['customerID', 'MonthlyCharges', 'TotalCharges', 'Churn']
    df_keep = df[list_drop]
    df_test = df.drop(columns=list_drop)

    categorical_columns = df_test.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_columns:
        if df_test[col].dtype == 'object':

            df_test[col] = le.fit_transform(df_test[col])

    df = pd.concat([df_keep, df_test], axis=1)
    df['Churn'] = le.fit_transform(df['Churn'])

    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]

    df['tenure_bin'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=True)
    df['tenure_bin'] = le.fit_transform(df['tenure_bin'])

    return df


def cleaning_table(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    space_count = df.map(lambda x: str(x).count(' '))
    total_spaces = space_count.sum().sum()

    if 0 <total_spaces < 100:
        filtered_df =df.apply(lambda row: any(' ' in str(cell) for cell in row), axis=1)
        df = df[~filtered_df]

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    median_total = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_total)

    numeric_columns = ['MonthlyCharges', 'TotalCharges']
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df


data_churn = pd.read_csv('files/dataset.csv')
df_churn = handle_categorical_values(data_churn)
df_churn = cleaning_table(df_churn)
churn_analysis = ChurnAnalysis(data_path=df_churn)
churn_analysis.run_analysis()
