"""
Description: 
his script loads cleaned datasets, standardizes the data, trains and evaluates multiple machine learning models, and selects the top two models based on accuracy.
"""
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from typing import List, Any
from sklearn.model_selection import cross_val_score

from src.data_collection_preprocessing import load_data

# standardization is used when the features do not have a Gaussian distribution.

def standardize_data(data: pd.DataFrame, cols: List[str], scaler: StandardScaler = None) -> pd.DataFrame:
    """
    Standardize the specified columns in the input data.
    
    Parameters:
    - data (pd.DataFrame): Input data.
    - cols (List[str]): List of column names to be standardized.
    - scaler (StandardScaler): StandardScaler object with parameters from training set.
    
    Returns:
    - pd.DataFrame: Standardized data.
    """
    if not scaler:
        scaler = StandardScaler()
        data[cols] = scaler.fit_transform(data[cols])
    else:
        data[cols] = scaler.transform(data[cols])
    return data, scaler

def model_prediction_evaluation(models: List, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """
    Trains and evaluates multiple machine learning models.

    Parameters:
    - models (List): A list of machine learning models to be trained and evaluated.
    - X_train (pd.DataFrame): The training data features.
    - y_train (pd.Series): The training data labels.
    - X_val (pd.DataFrame): The validation data features.
    - y_val (pd.Series): The validation data labels.

    Returns:
    - model_eval (dict): A dictionary containing the evaluation results for each model.
                        The keys are the model names and the values are dictionaries
                        with the 'accuracy' metric for each model.
    """
    model_eval = {}
    for name , model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred) #Overall correctness of the model.
        model_eval[name] = {'accuracy': accuracy}
        confusion_matrix_plot(y_val, y_pred, model_name=name)
    return model_eval

def create_model(model_name: str, best_params: dict = None):
    if model_name == 'SVC':
        if best_params:
            return SVC(**best_params)
        else:
            return SVC()
    elif model_name == 'LogisticRegression':
        if best_params:
            return LogisticRegression(**best_params)
        else:
            return LogisticRegression()
    elif model_name == 'RandomForestClassifier':
        if best_params:
            return RandomForestClassifier(**best_params)
        else:
            return RandomForestClassifier()
    elif model_name == 'GradientBoostingClassifier':
        if best_params:
            return GradientBoostingClassifier(**best_params)
        else:
            return GradientBoostingClassifier()
    elif model_name == 'KNeighborsClassifier':
        if best_params:
            return KNeighborsClassifier(**best_params)
        else:
            return KNeighborsClassifier()
    elif model_name == 'DecisionTreeClassifier':
        if best_params:
            return DecisionTreeClassifier(**best_params)
        else:
            return DecisionTreeClassifier()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def confusion_matrix_plot(y_test: pd.Series, y_pred: pd.Series, model_name: str) -> None:
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'plots/Confusion_Matrix_{model_name}.png')

def cross_validate_model(tuned_model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:   
    tuned_model.fit(X_train, y_train)
    # Perform 10-fold cross-validation on the training set
    cv_scores = cross_val_score(tuned_model, X_train, y_train, cv=10, scoring='accuracy')

    print("Cross-validation scores:", cv_scores)
    print("Average cross-validation score:", np.mean(cv_scores))

    y_pred = tuned_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    print("Test set accuracy :", test_accuracy)


def save_model(model: Any, file_path: str) -> None:
    joblib.dump(model, file_path)

if __name__ == '__main__':

    # As per feature importance, the following features are important for predicting customer churn: Monthly Charges, Annual Income, Age, Years with Company, Number of Support Calls
    col_names = ['Monthly Charges','Annual Income','Age', 'Years with Company','Number of Support Calls', 'Gender', 'Churn', 'Income_Level', 'Age Group', 'Loyalty_Level']
    train_fpath = 'data/cleaned_train_dataset.csv'
    train_data = load_data(train_fpath, col_names)
    # standardize the data and Split the data into features and target variable
    X_train, scaler = standardize_data(train_data, ['Monthly Charges','Annual Income','Age', 'Years with Company','Number of Support Calls','Income_Level', 'Age Group', 'Loyalty_Level'])
    y_train = train_data['Churn']

    val_fpath = 'data\cleaned_validation_dataset.csv'
    val_data = load_data(val_fpath, col_names)
    X_val, _ = standardize_data(val_data, ['Monthly Charges','Annual Income','Age', 'Years with Company','Number of Support Calls','Income_Level', 'Age Group', 'Loyalty_Level'], scaler=scaler)
    y_val = val_data['Churn']

    models = [
        ['KNeighborsClassifier', KNeighborsClassifier()],
    ]

    metrics = model_prediction_evaluation(models, X_train, y_train, X_val, y_val)
    # Find the first two models with the highest accuracy
    sorted_models = sorted(metrics.keys(), key=lambda x: metrics[x]['accuracy'], reverse=True)[:3]

    print('models evaluation metrics : Train and Val set \n', metrics)
    print(f"The best models  are {', '.join(sorted_models)} with accuracies of {metrics[sorted_models[0]]['accuracy']}")

