# Telco Customer Churn Prediction
## Project Description
This project aims to analyze the churn behavior of customers in a telecommunications company using a dataset from Kaggle and develop a predictive model to identify whether a customer will leave the service or not.

## Project steps
- Exploratory Data Analysis (EDA)
- Data Visualization
- Feature Engineering
- Testing multiple Machine Learning algorithms
- Comparing evaluation metrics across models

## Goal
- Understand the key factors influencing customer churn.
- Build an accurate model to predict churn.
- Identify the best-performing algorithm based on metrics such as accuracy, F1-score, recall, and precision.

## Dataset
The dataset used is the Telco Customer Churn Dataset, available on Kaggle. It contains demographic, contractual, and behavioral data about customers.

Dataset Details
Rows: XXXX (e.g., 7,043 records)
Columns: XX (e.g., 21 attributes)
Target: Churn (binary: "Yes" for churn, "No" for non-churn)
Examples of Features:
Demographic data: Age, Gender, Marital Status.
Contract data: Contract Type, Payment Method.
Usage data: Internet usage, Phone lines.
Project Workflow
1. Exploratory Data Analysis (EDA)
Studied the distribution of variables.
Identified missing values and handled outliers.
Analyzed correlations between features and the target variable (Churn).
2. Data Visualization
Created charts to better understand customer behavior:
Churn distribution across contract-related variables (e.g., contract type, payment method).
Relationships between numerical features (e.g., contract duration and churn).
3. Feature Engineering
Transformed categorical variables into numerical ones (e.g., encoding).
Created new features (e.g., relative duration based on contract type).
Standardized and scaled numerical data.
4. Machine Learning Models
The following algorithms were tested:

Logistic Regression
Random Forest
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Gradient Boosting (XGBoost, LightGBM)
AdaBoost
Each model was evaluated using the following metrics:

Accuracy
F1-Score
Recall
Precision
5. Model Comparison
The results were summarized in a comparison table to identify the most performant model.

## Requirements
To run the project, ensure you have the following installed:
- Python 3.7 or higher
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
Install dependencies with:
```pip install -r requirements.txt```
## Steps
- Clone the repository:
```git clone https://github.com/raphsa/Telco_Customer_churn.git```
- Activate the virtual environment:
python -m venv name_env
```source name_env/bin/activate```
- Install the file requirements.txt
- Run the main script:
```python main.py```
Results (charts and reports) will be saved in the ./images folder.
## License
This project is licensed under the MIT License. See the LICENSE file for details.