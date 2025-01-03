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
        ax = axes[i%2]  # set 1x2 grid
        churn_df[churn_df["Churn"]=="No"][var].plot(kind="hist", ax=ax, density=True, 
                                                       alpha=0.5, color="green", label="No")
        churn_df[churn_df["Churn"]=="Yes"][var].plot(kind="hist", ax=ax, density=True,
                                                        alpha=0.5, color="red", label="Yes")
        ax.set_title(f"Churn distribution for {var}")
        ax.set_ylabel("Frequence")
        ax.set_xlabel(var)
        ax.legend(title="Churn")
        handles, labels = ax.get_legend_handles_labels()
        new_labels = ["Sì" if label == "Yes" else label for label in labels]
        ax.legend(handles, new_labels)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
plt.tight_layout(pad=3.0)
plt.savefig("images/Churn_tenure.png")
plt.show()
# making plot to evaluate the churn distribution in the dataset
counts = churn_df["Churn"].value_counts()
# building the barplot with specified colours
colors = ["green" if label == "Yes" else "red" for label in counts.index]
counts.plot(kind="bar", color=colors)
# adding title and labels
plt.title("Distribution based on phone operator changes")
plt.xlabel("Churn")
plt.ylabel("Total clients")
plt.xticks(rotation=0)
plt.savefig("images/Churn_ds_dist.png")
plt.show()
# making plots on categories distribution between clients left and not left
def percentage_calc(var, target, df=churn_df):
    counts = df.groupby([var, target]).size().unstack(fill_value=0)
    percent = counts.div(counts.sum(axis=1), axis=0) * 100
    return percent
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
demog_var = ["Gender", "SeniorCitizen", "Partner", "Dependents"]
# developing method to insert plots in grid
def grid_plots(variables):
    for i, var in enumerate(variables):
        ax = axes[i//2, i%2]  # set a 2x2 grid
        percent = percentage_calc(var, "Churn")
        percent.plot(kind="bar", stacked=True, color=["green","red"], ax=ax)
        ax.set_title(f"Churn distribution for {var}")
        ax.set_ylim(0, 100) 
        ax.set_ylabel("Percentage")
        ax.set_xlabel(var)
        ax.legend(title="Churn")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        handles, labels = ax.get_legend_handles_labels()
        new_labels = ["Sì" if label == "Yes" else label for label in labels]
        ax.legend(handles, new_labels)
grid_plots(demog_var)
axes[0,1].set_xticklabels(["No","Yes"], rotation=0)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
plt.tight_layout(pad=3.0)
plt.savefig("images/demog_churn_dist.png")
plt.show()
# making same plots for other connection related variables
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
connection_var = ["PhoneService","MultipleLines","InternetService","StreamingTV"]
grid_plots(connection_var)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
plt.tight_layout(pad=3.0)
plt.savefig("images/connect_churn_dist.png")
plt.show()
# making same plots for client information related variables
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
customer_var = ["Contract","PaperlessBilling","PaymentMethod"]
grid_plots(customer_var)
axes[1,1].axis("off")
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
plt.tight_layout(pad=3.0)
plt.savefig("images/infoclient_churn_dist.png")
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
plt.barh(variables_sorted, mi_values_sorted, color="skyblue")
plt.title("Variables Mutual Information")
plt.ylabel("Variables")
plt.xlabel("Mutual Information")
plt.tight_layout(pad=3.0)
plt.savefig("images/mutual_info.png")
plt.show()

## Feature engineering
# things done:
# 1 dicotomic variables generation
# 2 one-hot-encoding for categorical variables with more than two labels
# 3 normalization of quantitative variables
# dicotomic variables generation
churn_df_mod = churn_df.copy()
churn_df_mod.Gender = churn_df.Gender.map({"Female":1,"Male":0})
yes_no_columns = ["Partner","Dependents","PhoneService","PaperlessBilling","Churn"]
for i in yes_no_columns:
     churn_df_mod[i] = churn_df[i].map({"Yes":1,"No":0})
# one-hot-encoding
ohe_columns = ["MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaymentMethod"]
churn_df_mod = pd.get_dummies(churn_df_mod, columns = ohe_columns)
# normalization
norm_columns = ["Tenure","MonthlyCharges","TotalCharges"]
for i in norm_columns:
     min_col = churn_df_mod[i].min()
     max_col = churn_df_mod[i].max()
     churn_df_mod[i] = (churn_df_mod[i] - min_col)/(max_col-min_col)

## Model generation
X = churn_df_mod.drop(columns="Churn")
y = churn_df_mod.Churn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle = True)
# tested models:
# 1 KNN
# 2 GradientBoosting
# 3 AdaBoosting
# 4 SVM
# 5 Logistic Regression
# 6 Random Forest
models = ["KNN", "Gradient Boosting", "AdaBoost","SVM","Logistic Regression","Random Forest"]

def func_metrics(y_test, y_pred, metrics, model):
    
    accuracy = round(accuracy_score(y_test,y_pred),3)
    precision = round(precision_score(y_test, y_pred),3) 
    recall = round(recall_score(y_test, y_pred),3)
    f1 = round(f1_score(y_test, y_pred),3)
    dict_met = {"Model": [model],
                "Accuracy score": [accuracy],
                "Precision score": [precision],
                "Recall score": [recall],
                "F1 score":[f1]}
    
    metric = pd.DataFrame(data=dict_met)
    metrics = pd.concat([metrics,metric])
    
    return metrics

def func_models(X_train, X_test, y_train, y_test, models):
     metrics = pd.DataFrame()
     opt_params = []
     for mod in models:
        if mod=="KNN":
            param_grid = {"n_neighbors": [2,3,5,10],
                         "p": [1,2]}
            knn_model = KNeighborsClassifier()
            grid = GridSearchCV(knn_model, param_grid)
            grid.fit(X_train, y_train)
            model = KNeighborsClassifier(**grid.best_params_)
            opt_params.append(grid.best_params_)

        elif mod=="SVM":
            param_grid = {"C": [0.001,0.01,0.1,0.5,1,2,5],
                         "kernel": ["linear","rbf","poly"],
                         "gamma": ["scale","auto"],
                         "degree": [2,3,4]}
            svc_model = SVC()
            grid = GridSearchCV(svc_model, param_grid)
            grid.fit(X_train, y_train)
            model = SVC(**grid.best_params_)
            opt_params.append(grid.best_params_)

        elif mod=="Random Forest":
            param_grid = {"n_estimators": [15,25,50,64,100,200],
                         "max_features": [2,3,5],
                         "bootstrap": [True,False]}
            rfc = RandomForestClassifier()
            grid = GridSearchCV(rfc, param_grid)
            grid.fit(X_train, y_train)
            model = RandomForestClassifier(**grid.best_params_)
            opt_params.append(grid.best_params_)

        elif mod=="AdaBoost":
            param_grid = {"n_estimators": [5,10,25,50,100],
                         "learning_rate": [0.01,0.05,0.1,0.25,0.5]}
            ada_model = AdaBoostClassifier()
            grid = GridSearchCV(ada_model, param_grid)
            grid.fit(X_train, y_train)
            model = AdaBoostClassifier(**grid.best_params_)
            opt_params.append(grid.best_params_)

        elif mod=="Gradient Boosting":
            param_grid = {"n_estimators": [10,25,50],
                         "learning_rate": [0.01,0.05,0.1,0.5],
                         "max_depth": [3,4,5]}
            gb_model = GradientBoostingClassifier()
            grid = GridSearchCV(gb_model, param_grid)
            grid.fit(X_train, y_train)
            model = GradientBoostingClassifier(**grid.best_params_)
            opt_params.append(grid.best_params_)

        elif mod=="Logistic Regression":
            param_grid = {"C":[0.001,0.1,1,5],
                          "solver":["lbfgs","liblinear","saga"],
                          "max_iter":[500]}
            logreg_model = LogisticRegression()
            grid = GridSearchCV(logreg_model, param_grid)
            grid.fit(X_train, y_train)
            model = LogisticRegression(**grid.best_params_)
            opt_params.append(grid.best_params_)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = func_metrics(y_test, y_pred, metrics, mod)
     return metrics, opt_params

metrics, opt_params = func_models(X_train, X_test, y_train, y_test, models)
# plotting table with algorithms results
fig, ax = plt.subplots()
# hiding axes
ax.axis("tight")
ax.axis("off")
table = ax.table(cellText=metrics.values, colLabels=metrics.columns, cellLoc="center", loc="center")
plt.savefig("images/model_performances.png")
plt.show()

print(metrics)
print(opt_params)
## I pick the SVM because, having the same Accuracy, it has an higher F1 score and most of all an higher Recall
# I prefer looking at Recall and Precision because an higher Recall means we better predict which clients will unsubscript, in order to focus on building customer loyalty
# The best SVM is with a linear kernel, C=5, degree=2 and gamma=scale
