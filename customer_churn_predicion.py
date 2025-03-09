import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import model_selection
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
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

    numeric_columns = df_keep.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df_test.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_columns:
        if df_test[col].dtype == 'object':
            # encoded_cols = pd.get_dummies(df_test[col], prefix=col)
            # df_test = pd.concat([df_test.drop(col, axis=1), encoded_cols], axis=1)

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


def split_train_test_model(df,  test_size=0.2, random_state=42):
    X = df.drop(columns=["Churn", 'TotalCharges', 'tenure'])
    X = X.set_index('customerID')

    y = df[['Churn', 'customerID']]
    y = y.set_index('customerID')

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


class ChurnModelEvaluator:
    def __init__(self, X_train, y_train, X_test, y_test, graphs_dir="graph", results_dir="result"):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.graphs_dir = graphs_dir
        self.results_dir = results_dir
        os.makedirs(self.graphs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        self.tuned_models = {}

    def tune_gradient_boosting(self, use_randomized=False):
        if use_randomized:
            param_dist = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6],
                'subsample': [0.6, 0.8, 1.0]
            }
            tuner = RandomizedSearchCV(
                GradientBoostingClassifier(random_state=42),
                param_distributions=param_dist,
                n_iter=20,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='accuracy',
                random_state=42,
                n_jobs=-1
            )
        else:
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
            tuner = GridSearchCV(
                GradientBoostingClassifier(random_state=42),
                param_grid=param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
        tuner.fit(self.X_train, self.y_train)
        best_model = tuner.best_estimator_
        self.tuned_models["GradientBoostingClassifier"] = best_model
        print("Best GradientBoostingClassifier params:", tuner.best_params_)
        return best_model

    def tune_adaboost(self):
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]
        }
        tuner = GridSearchCV(
            AdaBoostClassifier(random_state=42),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        tuner.fit(self.X_train, self.y_train)
        best_model = tuner.best_estimator_
        self.tuned_models["AdaBoostClassifier"] = best_model
        print("Best AdaBoostClassifier params:", tuner.best_params_)
        return best_model

    def tune_ridge(self):
        param_grid = {
            'alphas': [[0.1, 1.0, 10.0, 20.0]]
        }
        tuner = GridSearchCV(
            RidgeClassifierCV(),
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        tuner.fit(self.X_train, self.y_train)
        best_model = tuner.best_estimator_
        self.tuned_models["LogisticRegressionCV"] = best_model
        print("Best RidgeClassifierCV params:", tuner.best_params_)
        return best_model

    def evaluate_and_save(self, model, model_name):
        y_pred = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred) * 100
        print(f"\nModel: {model_name}")
        print("Accuracy: {:.2f}%".format(acc))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

        # Save confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Confusion Matrix: {model_name}")
        cm_path = os.path.join(self.graphs_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        # Save feature importance or coefficients if available
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = self.X_train.columns
            feat_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            print("\nFeature Importances for", model_name)
            print(feat_imp_df)
            csv_path = os.path.join(self.results_dir, f"{model_name}_feature_importances.csv")
            feat_imp_df.to_csv(csv_path, index=False)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feat_imp_df, x='Importance', y='Feature')
            plt.title(f"Feature Importances: {model_name}")
            plt.tight_layout()
            feat_imp_path = os.path.join(self.graphs_dir, f"{model_name}_feature_importances.png")
            plt.savefig(feat_imp_path)
            plt.close()
        elif hasattr(model, "coef_"):
            coefs = model.coef_.ravel()
            importances = np.abs(coefs)
            feature_names = self.X_train.columns
            feat_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefs,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            print("\nCoefficients for", model_name)
            print(feat_imp_df)
            csv_path = os.path.join(self.results_dir, f"{model_name}_coefficients.csv")
            feat_imp_df.to_csv(csv_path, index=False)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feat_imp_df, x='Importance', y='Feature')
            plt.title(f"Feature Coefficients (Importance): {model_name}")
            plt.tight_layout()
            feat_imp_path = os.path.join(self.graphs_dir, f"{model_name}_coefficients.png")
            plt.savefig(feat_imp_path)
            plt.close()
        else:
            print(f"No feature importance or coefficients available for {model_name}.")

    def evaluate_all(self):
        for name, model in self.tuned_models.items():
            self.evaluate_and_save(model, name)

    def build_voting_classifier(self):
        required_models = {"GradientBoostingClassifier", "AdaBoostClassifier", "LogisticRegressionCV"}
        if not required_models.issubset(set(self.tuned_models.keys())):
            print("Please tune GradientBoosting, AdaBoost, and LogisticRegressionCV models first.")
            return None

        voting_clf = VotingClassifier(
            estimators=[
                ('gb', self.tuned_models["GradientBoostingClassifier"]),
                ('ada', self.tuned_models["AdaBoostClassifier"]),
                ('lgcv', self.tuned_models["LogisticRegressionCV"])
            ],
            voting='hard'
        )
        voting_clf.fit(self.X_train, self.y_train)

        y_pred = voting_clf.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred) * 100
        print("Voting Classifier Accuracy: {:.2f}%".format(acc))
        print("Classification Report (Voting):\n", classification_report(self.y_test, y_pred))

        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix: VotingClassifier")
        voting_cm_path = os.path.join(self.graphs_dir, "VotingClassifier_confusion_matrix.png")
        plt.savefig(voting_cm_path)
        plt.close()

        importances_list = []
        model_names = []
        for name in required_models:
            model = self.tuned_models[name]
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                importances_list.append(imp)
                model_names.append(name)
            elif hasattr(model, "coef_"):
                imp = np.abs(model.coef_).ravel()
                importances_list.append(imp)
                model_names.append(name)

        if importances_list:
            avg_importance = np.mean(np.array(importances_list), axis=0)
            feature_names = self.X_train.columns
            feat_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'AvgImportance': avg_importance
            }).sort_values(by='AvgImportance', ascending=False)
            print("\nAggregated Feature Importances (VotingClassifier):")
            print(feat_imp_df)

            csv_path = os.path.join("result", "VotingClassifier_feature_importances.csv")
            feat_imp_df.to_csv(csv_path, index=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=feat_imp_df, x='AvgImportance', y='Feature')
            plt.title("Aggregated Feature Importances (VotingClassifier)")
            plt.tight_layout()
            feat_imp_path = os.path.join(self.graphs_dir, "VotingClassifier_feature_importances.png")
            plt.savefig(feat_imp_path)
            plt.close()
        else:
            print("No feature importance or coefficients available for the constituent models.")

        return voting_clf

    def get_final_dataset_with_predictions(self, model, output_filename="final_dataset_with_predictions.csv"):
        y_pred = model.predict(self.X_test)

        if "customerID" not in self.X_test.columns:
            self.X_test = self.X_test.reset_index()

        if isinstance(self.y_test, np.ndarray):
            self.y_test = pd.Series(self.y_test, index=self.X_test.index, name="ActualChurn")

        actual_df = self.y_test.reset_index()

        if "customerID" in self.X_test.columns and "customerID" in actual_df.columns:
            final_df = self.X_test.merge(actual_df, on="customerID", how="left")
        else:
            final_df = self.X_test.copy()
            final_df["ActualChurn"] = self.y_test.values

        final_df["PredictedChurn"] = y_pred

        output_path = os.path.join(self.results_dir, output_filename)
        final_df.to_csv(output_path, index=False)
        print(f"Final dataset with predictions saved to: {output_path}")

        return final_df


data_churn = pd.read_csv('files/dataset.csv')
print_dataframe_stats(data_churn)

df_churn = handle_categorical_values(data_churn)
df_churn = cleaning_table(df_churn)
df_churn.to_csv('result/df_churn.csv', index=False)
df_churn_tuned = df_churn[['customerID',
                           'MonthlyCharges',
                           'TotalCharges',
                           'Churn',
                           'Contract',
                           'tenure_bin',
                           'OnlineSecurity',
                           'TechSupport',
                           # 'PhoneService',
                           # 'InternetService',
                           'tenure'
                           ]]

X, y, X_train, X_test, y_train, y_test = split_train_test_model(df_churn_tuned)
model_metrics_all = InitializingModels().get_model_results(X, y)

evaluator = ChurnModelEvaluator(X_train, y_train, X_test, y_test, graphs_dir="graph", results_dir="result")

gb_model = evaluator.tune_gradient_boosting(use_randomized=False)
ab_model = evaluator.tune_adaboost()
rc_model = evaluator.tune_ridge()

evaluator.evaluate_all()

voting_model = evaluator.build_voting_classifier()

final_dataset = evaluator.get_final_dataset_with_predictions(voting_model)
