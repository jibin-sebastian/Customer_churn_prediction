import pandas as pd
from typing import Tuple
import scipy
from sklearn.model_selection import train_test_split

def load_data(file_path: str, col_names: list = None) -> pd.DataFrame:
    if col_names:
        return pd.read_csv(file_path, usecols=col_names)
    else:
        return pd.read_csv(file_path)

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna()

def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    # Binning Annual Income
    data['Income_Level'] = pd.cut(data['Annual Income'], bins=[0, 50000, 90000, float('inf')], labels=['Low', 'Medium', 'High'])
    # Create a new feature 'Age Group'
    data['Age Group'] = pd.cut(data['Age'], bins=[0, 30, 60, float('inf')], labels=['Young', 'Adult', 'Senior'])
    # Binning Years with Company
    data['Loyalty_Level'] = pd.cut(data['Years with Company'], bins=[0, 2, 5, float('inf')], labels=['Low', 'Medium', 'High'])

    # implement label encoding for Gender and Churn columns
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    data['Income_Level'] = data['Income_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
    data['Age Group'] = data['Age Group'].map({'Young': 0, 'Adult': 1, 'Senior': 2})
    data['Loyalty_Level'] = data['Loyalty_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})

    return data

def drop_columns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drop the specified columns from the data.
    
    Parameters:
    - data (pd.DataFrame): Input data.
    - columns (list): List of columns to drop.
    
    Returns:
    - pd.DataFrame: Data with dropped columns.
    """
    return data.drop(columns, axis=1)

def save_data(data: pd.DataFrame, file_path: str) -> None:
    """
    Save the cleaned data to a CSV file.
    
    Parameters:
    - data (pd.DataFrame): Cleaned data.
    - file_path (str): Path to save the CSV file.
    """
    data.to_csv(file_path, index=False)

def split_data(data: pd.DataFrame, size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training, validation, and test sets.
    
    Parameters:
    - data (pd.DataFrame): Input data.
    - test_size (float): Proportion of the data to include in the test split.
    - random_state (int): Seed for the random number generator.
    
    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation, and test sets.
    """
    train_data, temp_data = train_test_split(data, test_size=size, shuffle=True, random_state=42)
    validation_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=True, random_state=42)
    return train_data, validation_data, test_data

if __name__ == "__main__":
    data = load_data("data\customer_churn_dataset_large.csv")

    train_data, validation_data, test_data = split_data(data, size=0.2)

    # perform data preprocessing , feature engineering and one-hot encoding for training dataset
    train_data = clean_data(train_data)
    train_data = feature_engineering(train_data)

    train_data = drop_columns(train_data, ['CustomerID'])
    save_data(train_data, "data\cleaned_train_dataset.csv")
    print('train dataset :',train_data.shape)

    # perform data preprocessing , feature engineering and one-hot encoding for validation dataset
    validation_data = clean_data(validation_data)
    validation_data = feature_engineering(validation_data)

    validation_data = drop_columns(validation_data, ['CustomerID'])
    save_data(validation_data, "data\cleaned_validation_dataset.csv")
    print('validation data :',validation_data.shape)

    # perform data preprocessing , feature engineering and one-hot encoding for test dataset
    test_data = clean_data(test_data)
    test_data = feature_engineering(test_data)
    
    test_data = drop_columns(test_data, ['CustomerID'])
    save_data(test_data, "data\\test_dataset.csv")
    print('test data :',test_data.shape)
