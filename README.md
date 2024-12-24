# Telco Customer Churn Prediction
## Project Description
This project aims to analyze the churn behavior of customers in a telecommunications company using a dataset from Kaggle and develop a predictive model to identify whether a customer will leave the service or not.

## Goal
- Understand the key factors influencing customer churn.
- Build an accurate model to predict churn.
- Identify the best-performing algorithm based on metrics such as accuracy, F1-score, recall, and precision.

## Dataset
The dataset used is the Telco Customer Churn Dataset, available on Kaggle at the following link:
<pre> https://www.kaggle.com/datasets/blastchar/telco-customer-churn </pre>
It contains demographic, contractual, and behavioral data about customers.
Dataset Details:
- Target variable: Churn (binary: "Yes" for churn, "No" for non-churn)
- Demographic variables: Age, Gender, Marital Status.
- Contract variables: Contract Type, Payment Method.
- Usage variables: Internet usage, Phone lines.

## Project steps
- Exploratory Data Analysis (EDA)
- Data Visualization
- Feature Engineering
- Testing multiple Machine Learning algorithms
- Comparing evaluation metrics across Machine Learning models
Models tested:
<pre> Logistic Regression </pre>
<pre> Random Forest </pre>
<pre> Support Vector Machine (SVM) </pre>
<pre> K-Nearest Neighbors (KNN) </pre>
<pre> Gradient Boosting (XGBoost) </pre>
<pre> AdaBoost </pre>
Each model was evaluated using the following metrics:
<pre> Accuracy </pre>
<pre> F1 Score </pre>
<pre> Recall </pre>
<pre> Precision </pre>

## Requirements
To run the project, ensure you have the following installed:
- Python 3.7 or higher
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
Install dependencies with:

```bash pip install -r requirements.txt ```

## Steps
- Clone the repository:

```bash git clone https://github.com/raphsa/Telco_Customer_churn.git ```

- Activate the virtual environment:

```bash python -m venv name_env /n source name_env/bin/activate ```

- Install the file ```requirements.txt```
- Run the main script:

```bash python main.py ```

Results (charts and reports) will be saved in the ./images folder.
## License
This project is licensed under the MIT License. See the LICENSE file for details.