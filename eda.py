import numpy as np
import pandas as pd
import scipy.stats as ss
import os
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


class ChurnAnalysis:
    def __init__(self, data_path, target_column='Churn', graphs_dir="graphs_eda", results_dir="results_eda"):
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

    def plot_pair_plots(self):
        """
        Plot and save pair plots for numeric columns with the target column as hue.
        """
        numeric_columns = self.df.select_dtypes(include=['int', 'float']).columns.tolist()

        pair_plot = sns.pairplot(self.df[numeric_columns], hue=self.target_column, diag_kind='hist')
        plot_path = os.path.join(self.graphs_dir, "pair_plots.png")
        pair_plot.fig.savefig(plot_path)
        plt.close(pair_plot.fig)
        print(f"Saved: {plot_path}")

    def plot_pair_plots_separate(self):
        """
        Plot and save separate scatter plots for each pair of numeric columns with the target column as hue.
        """
        numeric_columns = self.df.select_dtypes(include=['int', 'float']).columns.tolist()

        # Optionally, exclude the target column itself from the numeric columns
        numeric_columns = [col for col in numeric_columns if col != self.target_column]

        # Create a scatter plot for each pair of numeric columns
        for i in range(len(numeric_columns)):
            for j in range(i + 1, len(numeric_columns)):
                col_x = numeric_columns[i]
                col_y = numeric_columns[j]

                plt.figure(figsize=(6, 4))
                sns.scatterplot(data=self.df, x=col_x, y=col_y, hue=self.target_column, palette='Set1')
                plt.title(f'{col_x} vs. {col_y}')
                plt.legend(title=self.target_column, loc='best')
                plt.grid(True)

                # Save each figure with a unique filename
                plot_filename = f"pairplot_{col_x}_vs_{col_y}.png"
                plot_path = os.path.join(self.graphs_dir, plot_filename)
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
                print(f"Saved: {plot_path}")

    def plot_hist_plots_separate(self):
        """
        Plot and save separate 2D histogram plots for each pair of numeric columns.
        Each plot counts the number of observations in each bin, effectively showing item counts.
        """
        numeric_columns = self.df.select_dtypes(include=['int', 'float']).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col != self.target_column]

        for i in range(len(numeric_columns)):
            for j in range(i + 1, len(numeric_columns)):
                col_x = numeric_columns[i]
                col_y = numeric_columns[j]

                plt.figure(figsize=(6, 4))
                sns.histplot(data=self.df, x=col_x, y=col_y, bins=30, cbar=True)
                plt.title(f'{col_x} vs. {col_y} (Item Count)')
                plt.xlabel(col_x)
                plt.ylabel(col_y)
                plt.grid(True)

                # Save each figure with a unique filename
                plot_filename = f"histplot_{col_x}_vs_{col_y}_count.png"
                plot_path = os.path.join(self.graphs_dir, plot_filename)
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
                print(f"Saved: {plot_path}")

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

    def plot_all_churn_counts(self, bins=5):
        """
        For every column (except the target), plot and save a chart comparing the count of churn (0 vs 1).
        For categorical columns (or those with few unique values), group by the column directly.
        For numeric columns with many unique values, bin them first.
        """
        for col in self.df.columns:
            if col == self.target_column:
                continue

            plt.figure(figsize=(8, 5))

            # For categorical (or near-categorical) variables:
            if self.df[col].dtype == 'object' or self.df[col].nunique() <= 10:
                # Create a pivot table: rows=group values, columns=churn values (0/1)
                churn_counts = self.df.groupby(col)[self.target_column].value_counts().unstack().fillna(0)
                churn_counts.plot(kind='bar', stacked=False, color=['skyblue', 'salmon'])
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.title(f"Churn Count by {col}")
            else:
                # For numeric variables: first, bin the column into a fixed number of groups.
                df_temp = self.df[[col, self.target_column]].copy()
                df_temp["bin"] = pd.cut(df_temp[col], bins=bins)
                churn_counts = df_temp.groupby("bin")[self.target_column].value_counts().unstack().fillna(0)
                churn_counts.plot(kind='bar', stacked=False, color=['skyblue', 'salmon'])
                plt.xlabel(f"{col} (binned)")
                plt.ylabel("Count")
                plt.title(f"Churn Count by binned {col}")

            plt.grid(True)
            plot_path = os.path.join(self.graphs_dir, f"churn_counts_by_{col}.png")
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            print(f"Saved: {plot_path}")

    def plot_all_boxplots(self):
        """
        For every numeric variable (except the target), plot and save a box plot of that variable's distribution across churn groups.
        """
        # Get numeric columns and exclude the target column.
        numeric_cols = [col for col in self.df.select_dtypes(include=['int', 'float']).columns
                        if col != self.target_column]

        for col in numeric_cols:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=self.target_column, y=col, data=self.df, palette="Set2")
            plt.xlabel(self.target_column)
            plt.ylabel(col)
            plt.title(f"{col} Distribution by {self.target_column}")
            plt.grid(True)

            plot_path = os.path.join(self.graphs_dir, f"boxplot_{col}_by_{self.target_column}.png")
            plt.savefig(plot_path, bbox_inches='tight')
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
        self.plot_pair_plots()
        self.plot_pair_plots_separate()
        self.plot_hist_plots_separate()
        self.plot_all_boxplots()
        self.plot_all_churn_counts()

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
