import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from src.data_collection_preprocessing import load_data

# Data Summary
def data_summary(data: pd.DataFrame) -> None:
    print("\nData Summary:")
    print(data.info())
    data.describe().to_csv('data/summary_statistics.csv')

def plot_distribution(data: pd.Series, title: str, xlabel: str, save_name: str, color: str) -> None:
    """
    Plot the distribution of given data.
    
    Parameters:
    - data (pd.Series): Input data.
    - title (str): Plot title.
    - xlabel (str): X-axis label.
    - save_name (str): File name to save the plot.
    - color (str): Color for the plot.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(data, bins=30, kde=True, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_name)
    #plt.show()

def plot_age_distribution(data: pd.Series) -> None:
    """
    Plot the distribution of Age.
    
    Parameters:
    - data (pd.Series): Input data for Age.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, bins=30, kde=True, color='salmon')
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig('plots/Distribution_of_Age.png')
    plt.show()

def plot_feature_vs_churn(feature: pd.Series, churn: pd.Series, feature_name: str) -> None:
    """
    Plot a feature vs churn.
    
    Parameters:
    - feature (pd.Series): Feature data for each customer.
    - churn (pd.Series): Churn status for each customer (1 for churned, 0 for non-churned).
    - feature_name (str): Name of the feature to be displayed on the x-axis.
    """
    # Count the number of churned and non-churned customers for each feature value
    counts = pd.crosstab(feature, churn)

    # Plotting
    plt.figure(figsize=(25, 7))
    counts.plot(kind='bar', stacked=False, color=['#FF9999', '#66B2FF'], figsize=(10,6))
    
    plt.title(f'{feature_name} vs Churn')
    plt.xlabel(feature_name)
    plt.ylabel('Count')
    plt.legend(title='Churn', labels=['Non-Churned Customers', 'Churned Customers'])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'plots/{feature_name}.png')

# plot churn count by age range
def plot_churn_count_by_age(data: pd.DataFrame) -> None:
    """
    Plot the count of churn by age range.
    
    Parameters:
    - data (pd.DataFrame): Input data.
    """
    data['Age Range'] = pd.cut(data['Age'], bins=[0, 18, 30, 40, 50, 60, 70, 80, 90, 100], labels=['0-18', '19-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100'])
    fig = px.histogram(data, x='Age Range', color='Churn', title='Churn Count by Age Range')
    fig.update_xaxes(title='Age Range')
    fig.update_yaxes(title='Count')
    fig.write_html('plots/plot_Churn_Count_by_Age_Range.html')


def plot_pie_chart(data: pd.Series, title: str, filename: str) -> None:
    """
    Plot a pie chart for the distribution of a categorical variable.
    
    Parameters:
    - data (pd.Series): Input data.
    - title (str): Title of the pie chart.
    - filename (str): Filename to save the plot.
    """
    count = data.value_counts()
    # Create the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(count, labels=count.index, autopct='%1.1f%%', startangle=140)    
    plt.title(title)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')
    plt.savefig(filename)

def plot_churn_count_by_income_range(income: pd.Series, churn: pd.Series) -> None:
    """
    Plot the count of churns based on different salary ranges.

    Parameters:
    - income (pd.Series): Input data for income.
    - churn (pd.Series): Input data for churn (values should be 0 or 1).
    """
    # Define income ranges
    income_ranges = ['< 40k', '40k - 60k', '60k - 80k', '80k - 100k', '> 100k']

    # Categorize incomes into different ranges
    income_range = pd.cut(income,
                          bins=[0, 40000, 60000, 80000, 100000, float('inf')],
                          labels=income_ranges,
                          right=False)

    # Combine income range and churn data into a DataFrame
    df = pd.DataFrame({'Income Range': income_range, 'Churn': churn})

    # Count churns for each income range
    churn_count = df['Churn'].groupby(df['Income Range']).value_counts().unstack(fill_value=0).reindex(income_ranges).reset_index()

    # Create the bar plot using Plotly
    fig = px.bar(churn_count,
                 x='Income Range',
                 y=[0, 1],
                 title='Count of Churns Based on Salary Ranges',
                 labels={'Income Range': 'Income Range', 'value': 'Churn Count', 'variable': 'Churn'},
                 color='variable',
                 color_discrete_map={0: 'blue', 1: 'red'},
                 barmode='group')

    # Update layout
    fig.update_xaxes(title='Income Range')
    fig.update_yaxes(title='Churn Count')
    fig.update_layout(legend_title='Churned: 1, Not Churned :0', legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig.write_html('plots/plot_Churn_Count_by_Income_Range.html')

def analyze_income_and_loyalty(income: pd.Series, years_with_company: pd.Series) -> None:
    """
    Analyze the distribution of annual income and years with the company and plot the graphs.
    
    Parameters:
    - income (pd.Series): Input data for income.
    - years_with_company (pd.Series): Input data for years with the company.
    """
    # Data Exploration
    print("Summary Statistics:")
    print(pd.DataFrame({'Income': income, 'Years_with_Company': years_with_company}).describe())
    
    plt.figure(figsize=(15, 6))
    # Histogram for Income
    plt.subplot(1, 2, 1)
    plt.hist(income, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Income')
    plt.xlabel('Income')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Histogram for Years with Company
    plt.subplot(1, 2, 2)
    plt.hist(years_with_company, bins=30, color='salmon', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Years with Company')
    plt.xlabel('Years with Company')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('plots/Income_and_Loyalty_Distribution.png')

def boxplot_feature_vs_churn(feature: pd.Series, churn: pd.Series, feature_name: str) -> None:
    """
    Plot the age distribution vs churn using a boxplot.
    
    Parameters:
    - feature (pd.Series): Feature data for each customer.
    - churn (pd.Series): Churn status for each customer (1 for churned, 0 for non-churned).
    """
    data = pd.DataFrame({'Feature': feature, 'Churn': churn})
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y='Feature', data=data, hue='Churn', palette='viridis', legend=False)
    sns.boxplot(x='Churn', y='Feature', data=data, palette='viridis')
    plt.title('Age Distribution vs Churn')
    plt.xlabel('Churn : Customer Churn Status (0: Not churned, 1: Churned)')
    plt.ylabel('Feature')
    plt.savefig(f'plots/{feature_name}_Boxplot.png')

def correlation_heatmap(data: pd.DataFrame) -> None:
    """
    Generate a correlation heatmap for the numerical columns in the given DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing numerical columns.

    Returns:
    None
    """
    # Selecting only the numerical columns
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(20, 15))
    sns.heatmap(data[numerical_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig('plots/Correlation_Heatmap.png')

def check_outliers_boxplot(data) -> None:
    
    num_columns = data.shape[1]
    num_rows = (num_columns - 1) // 3 + 1
    plt.figure(figsize=(25, 5 * num_rows))  
    for i, column in enumerate(data.columns[:-1]):
        plt.subplot(num_rows, 3, i + 1)
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot of {column}')
    
    plt.tight_layout()
    plt.savefig('plots/Outliers_Boxplot.png')

def feature_importance(data: pd.DataFrame) -> None:
    """
    Plot the feature importance of the given data.
    
    Parameters:
    - data (pd.DataFrame): Input data.
    """
    X = data.drop(columns=['Churn'])
    y = data['Churn']
    
    rf = RandomForestClassifier()
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    features = X.columns
    feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    # Calculate percentage importance for each feature
    feature_importances['Percentage Importance'] = (feature_importances['Importance'] / feature_importances['Importance'].sum()) * 100
    print('Feature importance after sorting \n',feature_importances)
    
    # Plot the feature importances
    plt.figure(figsize=(20, 10))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_importances, palette='viridis', legend=False)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig('plots/Feature_Importance.png')

if __name__ == "__main__":
    data = load_data('data\cleaned_train_dataset.csv')
    data_summary(data)

    # Plot distributions
    plot_age_distribution(data['Age'])
    plot_distribution(data['Annual Income'], 'Distribution of Annual Income', 'Annual Income', 'plots/Annual_Income_Distribution.png', 'salmon')
    plot_distribution(data['Years with Company'], 'Distribution of Years with Company', 'Years with Company', 'plots/Years_with_Company_Distribution.png', 'lightgreen')
    plot_distribution(data['Number of Support Calls'], 'Distribution of Number of Support Calls', 'Number of Support Calls', 'plots/Support_Calls_Distribution.png', 'orange')
    plot_distribution(data['Monthly Charges'], 'Distribution of Monthly Charges', 'Monthly Charges', 'plots/Monthly_Charges_Distribution.png', 'purple')
    
    plot_churn_count_by_age(data)
    plot_pie_chart(data['Gender'], 'Gender Distribution', 'plots/plot_Gender_Distribution_Pie.png')
    plot_pie_chart(data['Churn'], 'Churn Rate', 'plots/plot_churn_rate.png')
    plot_churn_count_by_income_range(data['Annual Income'], data['Churn'])
    analyze_income_and_loyalty(income=data['Annual Income'], years_with_company=data['Years with Company'])
    
    correlation_heatmap(data)
    check_outliers_boxplot(data)
    data.drop(columns=['Age Range'], inplace=True)
    feature_importance(data)
    plot_feature_vs_churn(data['Number of Support Calls'], data['Churn'], 'calls_vs_churn')
    plot_feature_vs_churn(data['Years with Company'], data['Churn'], 'Years_vs_churn')
    boxplot_feature_vs_churn(data['Age'], data['Churn'], 'Age_vs_churn')    