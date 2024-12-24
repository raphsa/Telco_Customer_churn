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
for col in churn_df.columns:
    print(f"{col}: {churn_df[col].unique()}")
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

## Data visualization
# making plot to evaluate the churn distribution based on monthly charges and total months of subscription
numerical_var = ["Tenure", "MonthlyCharges"]
fig, axes = plt.subplots(1, 2, figsize=(12, 10))
for i, var in enumerate(numerical_var):
        ax = axes[i%2]  # Posiziono i grafici in una griglia 1x2
        churn_df[churn_df["Churn"]=="No"][var].plot(kind="hist", ax=ax, density=True, 
                                                       alpha=0.5, color="green", label="No")
        churn_df[churn_df["Churn"]=="Yes"][var].plot(kind="hist", ax=ax, density=True,
                                                        alpha=0.5, color="red", label="Yes")
        ax.set_title(f"Distribuzione di Abbandono per {var}")
        ax.set_ylabel("Frequenza")
        ax.set_xlabel(var)
        ax.legend(title="Abbandono")
        handles, labels = ax.get_legend_handles_labels()
        new_labels = ['Sì' if label == 'Yes' else label for label in labels]
        ax.legend(handles, new_labels)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
plt.tight_layout(pad=3.0)
plt.show()
# making plot to evaluate the churn distribution in the dataset
counts = churn_df["Churn"].value_counts()
# building the barplot with specified colours
colors = ["green" if label == "Yes" else "red" for label in counts.index]
counts.plot(kind='bar', color=colors)
# adding title and labels
plt.title("Distribuzione sulla base del cambio di operatore telefonico")
plt.xlabel("Abbandono")
plt.ylabel("Numero di utenti nel dataset")
plt.xticks(rotation=0)
plt.show()
# making plots on categories distribution between clients left and not left
def percentage_calc(var, target, df=churn_df):
    counts = df.groupby([var, target]).size().unstack(fill_value=0)
    percent = counts.div(counts.sum(axis=1), axis=0) * 100
    return percent
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
demog_var = ["Gender", "SeniorCitizen", "Partner", "Dependents"]
# developing method to insert plots in grid
def grid_plots(variabili):
    for i, var in enumerate(variabili):
        ax = axes[i//2, i%2]  # set a 2x2 grid
        percent = percentage_calc(var, "Churn")
        percent.plot(kind="bar", stacked=True, color=["green","red"], ax=ax)
        ax.set_title(f"Distribuzione di Abbandono per {var}")
        ax.set_ylim(0, 100) 
        ax.set_ylabel("Percentuale")
        ax.set_xlabel(var)
        ax.legend(title="Abbandono")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        handles, labels = ax.get_legend_handles_labels()
        new_labels = ['Sì' if label == 'Yes' else label for label in labels]
        ax.legend(handles, new_labels)
grid_plots(demog_var)
axes[0,1].set_xticklabels(["No","Yes"], rotation=0)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
plt.tight_layout(pad=3.0)
plt.show()
# making same plots for other connession related variables
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
connection_var = ["PhoneService","MultipleLines","InternetService","StreamingTV"]
grid_plots(connection_var)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
plt.tight_layout(pad=3.0)
plt.show()
# making same plots for client information related variables
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
customer_var = ["Contract","PaperlessBilling","PaymentMethod"]
grid_plots(customer_var)
axes[1,1].axis("off")
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
plt.tight_layout(pad=3.0)
plt.show()

## Feature importance
# dividing between x and y
x = churn_df.select_dtypes(include=object).drop("Churn", axis=1)
y = churn_df.Churn
# computing variable relevance
def compute_mutual_information(categorical_serie):
    return mutual_info_score(categorical_serie, y)
mi_scores = x.apply(compute_mutual_information).sort_values(ascending=False)
print(mi_scores)
# making plot
mi_values = x.apply(compute_mutual_information)
mi_values_sorted, variables_sorted = zip(*sorted(zip(mi_values, x.columns), reverse=True))
plt.figure(figsize=(8, 6))
plt.barh(variables_sorted, mi_values_sorted, color='skyblue')
plt.title('Mutual Information per variabile')
plt.ylabel('Variabili')
plt.xlabel('Mutual Information')
plt.tight_layout(pad=3.0)
plt.show()