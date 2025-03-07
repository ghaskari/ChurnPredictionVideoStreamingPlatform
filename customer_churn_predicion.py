import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import model_selection
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier


def print_dataframe_stats(df):
    print(f"Rows   : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")
    print("\nFeatures : \n", df.columns.tolist())
    print("\nUnique values : \n", df.nunique())
    print("\nMissing values Total : ", df.isnull().sum().sum())
    print("\nMissing values : \n", df.isnull().sum())
    print("\nType of values: \n", df.dtypes)


def plots_eda(df):
    count_col = []
    hist_col = []
    for column in df.columns:
        unique_value = df[column].nunique()
        if unique_value <= 20:
            count_col.append(column)
        else:
            hist_col.append(column)


    plt.figure(figsize=(15,40))
    plot_num = 1
    for col in count_col:
        plt.subplot(10,2,plot_num)
        sns.countplot(data=df, x=col)
        plot_num += 1
        plt.tight_layout()

    plt.figure(figsize=(15,40))
    plot_num = 1
    for col in hist_col:
        plt.subplot(10,2,plot_num)
        sns.histplot(data=df, x=col,bins=25)
        plot_num += 1
        plt.tight_layout()

    plt.figure(figsize=(15,40))
    plot_num = 1
    for col in count_col:
        if df[col].nunique() <= 8 and col != "Churn":
            plt.subplot(10,2,plot_num)
            sns.countplot(data=df, x=col, hue="Churn")
            plot_num += 1
            plt.tight_layout()


def handle_categorical_values(df):

    le = LabelEncoder()

    list_drop = ['customerID', 'MonthlyCharges', 'TotalCharges', 'Churn']
    df_keep = df[list_drop]
    df_test = df.drop(columns=list_drop)

    numeric_columns = df_keep.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df_test.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_columns:
        if df_test[col].dtype == 'object':
            encoded_cols = pd.get_dummies(df_test[col], prefix=col)
            df_test = pd.concat([df_test.drop(col, axis=1), encoded_cols], axis=1)

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

    df['TotalCharges'] = df['TotalCharges'].astype(float)

    numeric_columns = ['MonthlyCharges', 'TotalCharges']
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df


def split_train_test_model(df,  test_size=0.2, random_state=42):
    X = df.drop(columns=["Churn", 'MonthlyCharges', 'tenure'])
    X = X.set_index('customerID')
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=random_state)

    return  X, y, X_train, X_test, y_train, y_test


class InitializingModels:

    def __init__(self):

        self.models = [
            # Ensemble
            AdaBoostClassifier(),
            BaggingClassifier(),
            GradientBoostingClassifier(),
            RandomForestClassifier(),

            # Linear Models
            LogisticRegressionCV(),
            RidgeClassifierCV(),

            # Nearest Neighbour
            KNeighborsClassifier(),

            # XGBoost
            XGBClassifier()
        ]

        self.metrics_cols = ['model_name', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        self.scoring = ['accuracy','precision', 'recall', 'f1']

        self.model_name = []
        self.test_accuracy = []
        self.test_precision = []
        self.test_recall = []
        self.test_f1 = []



    def get_model_results(self, X_variable, y_variable):
        for model in self.models:
            cv_results = model_selection.cross_validate(model, X_variable, y_variable, cv=5,
                                                        scoring=self.scoring, return_train_score=True)
            self.model_name.append(model.__class__.__name__)
            self.test_accuracy.append(round(cv_results['test_accuracy'].mean(),3)*100)
            self.test_precision.append(round(cv_results['test_precision'].mean(),3)*100)
            self.test_recall.append(round(cv_results['test_recall'].mean(),3)*100)
            self.test_f1.append(round(cv_results['test_f1'].mean(),3)*100)

        metrics_data = [self.model_name, self.test_accuracy, self.test_precision, self.test_recall, self.test_f1]
        m = {n:m for n,m in zip(self.metrics_cols,metrics_data)}
        model_metrics = pd.DataFrame(m)
        model_metrics = model_metrics.sort_values('test_accuracy', ascending=False)
        model_metrics.to_csv('result/model_creation.csv')
        print(model_metrics)

        return model_metrics


data_churn = pd.read_csv('files/dataset.csv')
print_dataframe_stats(data_churn)

df_churn = handle_categorical_values(data_churn)
df_churn = cleaning_table(df_churn)
df_churn.to_csv('files/df_churn.csv', index=False)

X, y, X_train, X_test, y_train, y_test = split_train_test_model(df_churn)
model_metrics_all = InitializingModels().get_model_results(X, y)
