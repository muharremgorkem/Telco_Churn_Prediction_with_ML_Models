#############################################################
# Telco Customer Churn Prediction: Machine Learning Application
#############################################################

# The task at hand involves developing a machine learning model capable of
# predicting customers who are likely to churn from the company.

# Dataset Story
###############
# The dataset contains information about a fictional telecom company operating in California.
# The company provides home phone and internet services to 7043 customers during the third quarter.
# The dataset reveals which customers have decided to discontinue the services, which have remained
# loyal, and which have recently subscribed.

# Variables
############
# CustomerId: Customer ID
# SeniorCitizen: Whether the customer is a senior citizen (1, 0)
# Partner: Whether the customer has a partner (Yes, No)
# Dependents: Whether the customer has dependents (Yes, No)
# Tenure: Number of months the customer has stayed with the company
# PhoneService: Whether the customer has a phone service (Yes, No)
# MultipleLines: Whether the customer has multiple lines (Yes, No, No Phone Service)
# InternetService: Internet service provider for the customer (DSL, Fiber Optic, No)
# OnlineSecurity: Whether the customer has online security (Yes, No, No Internet Service)
# OnlineBackup: Whether the customer has online backup (Yes, No, No Internet Service)
# DeviceProtection: Whether the customer has device protection (Yes, No, No Internet Service)
# TechSupport: Whether the customer receives tech support (Yes, No, No Internet Service)
# StreamingTV: Whether the customer has streaming TV (Yes, No, No Internet Service)
# StreamingMovies: Whether the customer has streaming movies (Yes, No, No Internet Service)
# Contract: Contract term of the customer (Month-to-Month, One Year, Two Years)
# PaperlessBilling: Whether the customer has paperless billing (Yes, No)
# PaymentMethod: Payment method of the customer (Electronic Check, Mailed Check, Bank Transfer (Automatic), Credit Card (Automatic))
# MonthlyCharges: Monthly amount charged to the customer
# TotalCharges: Total amount charged to the customer
# Churn: Whether the customer has churned (Yes or No)

################################################
# TASK 1: EXPLORATORY DATA ANALYSIS (EDA)
################################################
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tabulate import tabulate

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 100)

import warnings
warnings.simplefilter("ignore")

df = pd.read_csv('Datasets/Telco-Customer-Churn.csv')
df.columns = [col.lower() for col in df.columns]


def check_df(dataframe, head=5):
    print('################# Columns ################# ')
    print(dataframe.columns)
    print('################# Types  ################# ')
    print(dataframe.dtypes)
    print('##################  Head ################# ')
    print(dataframe.head(head))
    print('#################  Shape ################# ')
    print(dataframe.shape)
    print('#################  NA ################# ')
    print(dataframe.isnull().sum())
    print('#################  Quantiles ################# ')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99]).T)

check_df(df)

# Capturing numerical and Ccategorical variables
#############################################################
def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Correcting data with incorrect types
######################################
empty_rows = df[df['totalcharges'] == ' '] # bos degerler var
df['totalcharges'] = df['totalcharges'].replace(' ', float('nan')).astype(float)
df['totalcharges'].isnull().sum()

df["churn"] = df["churn"].apply(lambda x : 1 if x == "Yes" else 0)

# Numerical and categorical variable distribution within the data
##################################################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Categorical:
##############
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    if df[col].dtypes == 'bool':
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)

# Numerical:
############
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot = True)

# Examining categorical variables in relation to the target variable
####################################################################
def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "churn", col)

# Kadın ve erkeklerde churn yüzdesi neredeyse eşit
# Partner ve dependents'i olan müşterilerin churn oranı daha düşük
# PhoneServise ve MultipleLines'da fark yok
# Fiber Optik İnternet Servislerinde kayıp oranı çok daha yüksek
# No OnlineSecurity , OnlineBackup ve TechSupport gibi hizmetleri olmayan müşterilerin churn oranı yüksek
# Bir veya iki yıllık sözleşmeli Müşterilere kıyasla, aylık aboneliği olan Müşterilerin daha büyük bir yüzdesi churn
# Kağıtsız faturalandırmaya sahip olanların churn oranı daha fazla
# ElectronicCheck PaymentMethod'a sahip müşteriler, diğer seçeneklere kıyasla platformdan daha fazla ayrılma eğiliminde
# Yaşlı müşterilerde churn yüzdesi daha yüksektir

# Examination of outliers
###########################################
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))
# --> tenure False
# monthlycharges False
# totalcharges False

# Examination of missing values
##########################################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)
# -->               n_miss  ratio
# totalcharges      11  0.160

##################################
# KORELASYON
##################################
df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True) #--> high corr between TotalChargers'in and monthly charges with tenure


################################################
# TASK 2: FEATURE ENGINEERING
################################################

# Performing necessary procedures for missing and outlier observations
######################################################################
# No outlier observations were found, so no action has been taken.

# 11 missing values in the 'totalcharges' variable have been imputed with the mean.

df['totalcharges'] = df['totalcharges'].fillna(df['totalcharges'].mean())

##################################
# BASE MODEL KURULUMU
##################################

dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["churn"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)

y = dff["churn"]
X = dff.drop(["churn", "customerid"], axis=1)

def evaluate_model(model, X, y):
    cv_results = cross_validate(model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    report = classification_report(y, cross_val_predict(model, X, y, cv=5))
    mean_accuracy = cv_results['test_accuracy'].mean()
    mean_f1 = cv_results['test_f1'].mean()
    mean_roc_auc = cv_results['test_roc_auc'].mean()
    mean_precision = cv_results['test_precision'].mean()
    mean_recall = cv_results['test_recall'].mean()
    return {
        "report": report,
        "accuracy_mean": mean_accuracy,
        "roc_auc_mean": mean_roc_auc,
        "recall_mean": mean_recall,
        "precision_mean": mean_precision,
        "f1_mean": mean_f1,
    }

# Define the models
models = [
    ('LR', LogisticRegression(random_state=17)),
    ('KNN', KNeighborsClassifier()),
    ("CART", DecisionTreeClassifier(random_state=17)),
    ("Random Forest", RandomForestClassifier(random_state=17)),
    ("Gradient Boosting", GradientBoostingClassifier(random_state=17)),
    ('XGBoost', XGBClassifier(random_state=17)),
    ('LGBM', LGBMClassifier(random_state=17)),
    ('CatBoost', CatBoostClassifier(random_state=17, verbose=False))
]

# Create an empty DataFrame
results_dff = pd.DataFrame(columns=["Model", "Accuracy","ROC AUC", "Recall", "Precision", "F1 Score", ])

# Evaluate each model and add results to DataFrame
results = []

# Evaluate each model and add results to list
for model_name, model in models:
    model_results = evaluate_model(model, X, y)
    results.append({
        "Model": model_name,
        "Accuracy": model_results["accuracy_mean"],
        "ROC AUC": model_results["roc_auc_mean"],
        "Recall": model_results["recall_mean"],
        "Precision": model_results["precision_mean"],
        "F1 Score": model_results["f1_mean"],
    })

results_dff = pd.DataFrame(results)

print(results_dff)

#Results
#               Model  Accuracy  ROC AUC  Recall  Precision  F1 Score
#0                 LR     0.805    0.843   0.540      0.662     0.595
#1                KNN     0.762    0.744   0.445      0.565     0.498
#2               CART     0.731    0.654   0.490      0.492     0.491
#3      Random Forest     0.790    0.824   0.477      0.643     0.547
#4  Gradient Boosting     0.805    0.845   0.530      0.669     0.591
#5            XGBoost     0.782    0.822   0.504      0.607     0.551
#6               LGBM     0.795    0.834   0.520      0.640     0.573
#7           CatBoost     0.798    0.840   0.502      0.656     0.569

######################################
# Creating new variables
#######################################

# Monthly charges ratio
df['monthlychargesratio'] = df['monthlycharges'] / df['totalcharges']

# Monthly expenditure groups
bins = [0, 40.000, 80.000, 120000]
labels = ['low', 'medium', 'high']
df['monthlychargesgroups'] = pd.cut(df['monthlycharges'], bins=bins, labels=labels)

# Contract duration and number of dependents
df['contractanddependents'] = df['contract'] + '_' + df['dependents']

# Total number of services for a customer
df['NEW_TotalServices'] = (df[['phoneservice', 'internetservice', 'onlinesecurity',
                                       'onlinebackup', 'deviceprotection', 'techsupport',
                                       'streamingtv', 'streamingmovies']]== 'Yes').sum(axis=1)
df.info()
# 1 and 2 years contracts shown as Engaged
df["NEW_Engaged"] = df["contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# Paymentmethod groups(auto or not)
df["NEW_FLAG_AutoPayment"] = df["paymentmethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# Price per service
df["NEW_AVG_Service_Fee"] = df["monthlycharges"] / (df['NEW_TotalServices'] + 1)
df.info()

# Encoding
###########

# Binary cols - Label encoder
#############################
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Rare encoding
################
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "churn", cat_cols)
# No need for rare encoding!


# One-hot encoder
#################################
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape

# Standardization for Logistic regression and KNN
############################################################
df_ = df.copy()

scaler = StandardScaler()
df_[num_cols] = scaler.fit_transform(df_[num_cols])

#######################################
# TASK 3: MODELLING
#######################################

# 3.1. Ensemble Models
############################################
y = df["churn"]
X = df.drop(["churn", "customerid"], axis=1)

def evaluate_model(model, X, y):
    cv_results = cross_validate(model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    report = classification_report(y, cross_val_predict(model, X, y, cv=5))
    mean_accuracy = cv_results['test_accuracy'].mean()
    mean_f1 = cv_results['test_f1'].mean()
    mean_roc_auc = cv_results['test_roc_auc'].mean()
    mean_precision = cv_results['test_precision'].mean()
    mean_recall = cv_results['test_recall'].mean()
    return {
        "report": report,
        "accuracy_mean": mean_accuracy,
        "roc_auc_mean": mean_roc_auc,
        "recall_mean": mean_recall,
        "precision_mean": mean_precision,
        "f1_mean": mean_f1,
    }

# Define the models
models = [
    ('LR', LogisticRegression(random_state=17)),
    ('KNN', KNeighborsClassifier()),
    ("CART", DecisionTreeClassifier(random_state=17)),
    ("Random Forest", RandomForestClassifier(random_state=17)),
    ("Gradient Boosting", GradientBoostingClassifier(random_state=17)),
    ('XGBoost', XGBClassifier(random_state=17)),
    ('LGBM', LGBMClassifier(random_state=17)),
    ('CatBoost', CatBoostClassifier(random_state=17, verbose=False))
]

# Create an empty DataFrame
results_df = pd.DataFrame(columns=["Model", "Accuracy","ROC AUC", "Recall", "Precision", "F1 Score", ])

# Evaluate each model and add results to DataFrame
results = []

# Evaluate each model and add results to list
for model_name, model in models:
    model_results = evaluate_model(model, X, y)
    results.append({
        "Model": model_name,
        "Accuracy": model_results["accuracy_mean"],
        "ROC AUC": model_results["roc_auc_mean"],
        "Recall": model_results["recall_mean"],
        "Precision": model_results["precision_mean"],
        "F1 Score": model_results["f1_mean"],
    })

results_df = pd.DataFrame(results)

print(results_df)

#-->               Model  Accuracy  ROC AUC  Recall  Precision  F1 Score
#0                 LR     0.804    0.843   0.534      0.663     0.591
#1                KNN     0.767    0.754   0.461      0.577     0.512
#2               CART     0.721    0.646   0.483      0.475     0.479
#3      Random Forest     0.785    0.825   0.482      0.623     0.543
#4  Gradient Boosting     0.801    0.845   0.516      0.661     0.579
#5            XGBoost     0.781    0.822   0.501      0.607     0.549
#6               LGBM     0.791    0.832   0.516      0.631     0.567
#7           CatBoost     0.799    0.839   0.513      0.656     0.575


#################################
# 4. HYPERPARAMETER OPTIMIZATION
#################################
#-- > CatBoost model is chosen for hyperparameter optimization

# 4.1. CatBoost
#################
y = df["churn"]
X = df.drop(["churn", "customerid"], axis=1)

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500], "learning_rate": [0.01, 0.1], "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=False).fit(X, y)

catboost_best_grid.best_params_
# --> Out[8]: {'depth': 6, 'iterations': 500, 'learning_rate': 0.01}
catboost_final = catboost_model.set_params(depth=6, learning_rate=0.01,iterations=500, random_state=17).fit(X, y)

cv_results_catboost = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

catboost_results = {'Accuracy': cv_results_catboost['test_accuracy'].mean(),
    'F1 Score': cv_results_catboost['test_f1'].mean(),
    'ROC AUC': cv_results_catboost['test_roc_auc'].mean()}

print(catboost_results)

# Feature importance for CatBoost
#################################
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(catboost_final, X)
























































































































