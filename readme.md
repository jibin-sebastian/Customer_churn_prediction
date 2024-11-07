# Predictive Modeling for Customer Churn

## Objective
Build a predictive model to identify customers who are likely to churn.

## Dataset
You are provided with a dataset containing information about customers. The dataset includes the following columns:

- **CustomerID**: Unique identifier for each customer
- **Age**: Age of the customer
- **Gender**: Gender of the customer (Male/Female)
- **Annual Income**: Annual income of the customer
- **Years with Company**: Number of years the customer has been with the company
- **Number of Support Calls**: Number of customer support calls made by the customer
- **Monthly Charges**: Monthly charges billed to the customer
- **Churn**: Whether the customer churned or not (Yes/No)

## Tasks

### Data Exploration
- Perform exploratory data analysis to understand the data's distribution, patterns, and relationships.
- Visualize the data using appropriate plots (e.g., histograms, box plots, correlation matrices).

### Data Preprocessing
- Handle missing values appropriately.
- Encode categorical variables (if any).
- Split the dataset into training and testing sets.

### Model Building
- Choose an appropriate machine learning algorithm for the predictive task.
- Train the model on the training dataset.
- Evaluate the model's performance on the testing dataset using appropriate metrics (e.g., accuracy, precision, recall, F1-score, ROC AUC).

### Model Tuning (Optional)
- Perform hyperparameter tuning to improve the model's performance.
- Validate the tuned model using cross-validation.

### Interpretation
- Interpret the model's results.
- Identify the most important features influencing the prediction.

### Recommendations
- Provide recommendations based on the insights gained from the model to reduce customer churn.

## How to run scripts
On Windows:
Create a virtual environment: 
```bash
python -m venv venv
.\venv\Scripts\activate

python src\data_collection_preprocessing.py
python src\explanatory_data_analysis.py
python src\ml_modeling.py
python src\hyperparameter_tunning.py
python src\model_evaluation.py
