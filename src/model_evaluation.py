import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import json

from typing import Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from src.ml_modeling import load_data, standardize_data, create_model, save_model, cross_validate_model,confusion_matrix_plot

def final_model_evaluation(model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    model.fit(X_train, y_train)
    y_pred_on_test = model.predict(X_test)
    
    # plot and save confusion matrix
    confusion_matrix_plot(y_test, y_pred_on_test, model_name= 'final_model')

    return {
        'accuracy': accuracy_score(y_test, y_pred_on_test),
        'precision': precision_score(y_test, y_pred_on_test),
        'recall': recall_score(y_test, y_pred_on_test),
        'f1': f1_score(y_test, y_pred_on_test)
    }

def load_json(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)

if __name__ == '__main__':
    # Load the cleaned datasets
    col_names = ['Monthly Charges','Annual Income','Age', 'Years with Company','Number of Support Calls', 'Gender', 'Churn','Income_Level', 'Age Group', 'Loyalty_Level']
    train_fpath = 'data\cleaned_train_dataset.csv'
    train_data = load_data(train_fpath, col_names)
    # standardize the data and Split the data into features and target variable
    X_train,scaler = standardize_data(train_data, ['Monthly Charges','Annual Income','Age', 'Years with Company','Number of Support Calls','Income_Level', 'Age Group', 'Loyalty_Level'])
    y_train = train_data['Churn']

    test_fpath = 'data\\test_dataset.csv'
    test_data = load_data(test_fpath, col_names)
    X_test, _ = standardize_data(test_data, ['Monthly Charges','Annual Income','Age', 'Years with Company','Number of Support Calls','Income_Level', 'Age Group', 'Loyalty_Level'], scaler=scaler)
    y_test = test_data['Churn']

    # load best models
    best_models = load_json('models/best_models_params.json')

    sorted_models = sorted(best_models, key=lambda x: x['best_scores'], reverse=True)
    print('Best model after tuning with validation dataset',sorted_models[0])
    ml = create_model(sorted_models[0]['model'],sorted_models[0]['best_params'])

    cross_validate_model(ml, X_train, y_train, X_test, y_test) # use training set for cross validation to validate the  model's performance and ensure it generalizes well to unseen data.
    final_metrics = final_model_evaluation(ml, X_train, y_train, X_test, y_test)
    print(f"Final Model {sorted_models[0]['model']} Evaluation on Test dataset:\n", final_metrics)
    save_model(ml, f"models/final_model_{sorted_models[0]['model']}.pkl")
