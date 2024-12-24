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