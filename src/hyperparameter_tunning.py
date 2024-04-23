import pandas as pd
import numpy as np
import yaml
import json

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data_collection_preprocessing import load_data
from src.ml_modeling import standardize_data, create_model, save_model, cross_validate_model

def Hyperparameters_tunning(algorithm: dict, X_val: pd.DataFrame, y_val: pd.Series) -> pd.DataFrame:
    scores = []
    for name, mp in algorithm.items():
        grid_search = GridSearchCV(mp['model'], param_grid = mp['param_grid'], cv=5, scoring='accuracy', n_jobs=1, verbose=2)
        grid_search.fit(X_val, y_val)
        scores.append({
            'model': name, 
            'best_scores': grid_search.best_score_,
            'best_params': grid_search.best_params_
            })
    return pd.DataFrame(scores, columns=['model', 'best_scores', 'best_params'])


if __name__ == '__main__':

    # Load the cleaned datasets
    col_names = ['Monthly Charges','Annual Income','Age', 'Years with Company','Number of Support Calls', 'Gender', 'Churn', 'Income_Level', 'Age Group', 'Loyalty_Level']
    train_fpath = 'data\cleaned_train_dataset.csv'
    train_data = load_data(train_fpath, col_names)
    # standardize the data and Split the data into features and target variable
    X_train, scaler = standardize_data(train_data, ['Monthly Charges','Annual Income','Age', 'Years with Company','Number of Support Calls','Income_Level', 'Age Group', 'Loyalty_Level'])
    y_train = train_data['Churn']

    val_fpath = 'data\cleaned_validation_dataset.csv'
    val_data = load_data(val_fpath, col_names)
    X_val, _ = standardize_data(val_data, ['Monthly Charges','Annual Income','Age', 'Years with Company','Number of Support Calls','Income_Level', 'Age Group', 'Loyalty_Level'], scaler=scaler)
    y_val = val_data['Churn']

    # Load the configuration file
    with open('config\hyperparam_tunning.yml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Create model instances
    if config['algorithms'] is None:
        raise ValueError('No algorithms found in the configuration file')
    for name, mp in config['algorithms'].items():
        mp['model'] = create_model(mp['model'])
    best_models_df = Hyperparameters_tunning(config['algorithms'], X_val, y_val)
    best_models_df.to_json('models/best_models_params.json', orient='records')
    