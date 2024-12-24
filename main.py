## Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier  
from sklearn.svm import SVC
from sklearn.metrics import mutual_info_score, accuracy_score, precision_score, recall_score, f1_score

## Dataset 
churn_df = pd.read_csv("Telco_customer_churn.csv")

## Exploratory analysis and data cleaning
churn_df.columns
churn_df.head()
churn_df.info()
# checking unique values
for colonna in churn_df.columns:
    print(f"{colonna}: {churn_df[colonna].unique()}")
# deleting unnecessaries columns
churn_df.drop(columns="customerID", inplace=True)
# converting TotalCharges dtype
churn_df.TotalCharges = pd.to_numeric(churn_df.TotalCharges, errors = "coerce")
# checking how many non numeric rows in TotalCharges
print(f"The {round(len(churn_df[churn_df.TotalCharges.isnull()])/len(churn_df)*100,2)}% of the whole dataset has an NA value on the TotalCharges column, so only {len(churn_df[churn_df.TotalCharges.isnull()])} rows")
# deleting non numeric rows as not relevant in frequence
churn_df.dropna(inplace=True)
# changing PaymentMethod labels to have more readable names
churn_df["PaymentMethod"] = churn_df["PaymentMethod"].str.replace("Bank transfer (automatic)", "Bank transfer", regex=False)
churn_df["PaymentMethod"] = churn_df["PaymentMethod"].str.replace("Credit card (automatic)", "Credit card", regex=False)
# set all columns with the capital letter
churn_df.rename(columns={"gender":"Gender","tenure":"Tenure"}, inplace=True)
