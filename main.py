import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# colab link : https://colab.research.google.com/drive/13xqokz7tlLfVi2qvRo-BqNPZhQMMCrtT?usp=sharing


# Load the "loan_old.csv" dataset
loan_old = pd.read_csv("loan_old.csv")

# b) Perform analysis on the dataset
# i) Check for missing values
missing_values = loan_old.isnull().sum()
print("Missing values:\n", missing_values)

# ii) Check the type of each feature
feature_types = loan_old.dtypes
print("\nData types:\n", feature_types)

# iii) Check whether numerical features have the same scale
numerical_features = loan_old.select_dtypes(include=[np.number]).columns
numerical_scales = loan_old[numerical_features].describe().loc[['mean', 'std']]
print("\nNumerical features summary:\n", numerical_scales)

# iv) Visualize a pairplot between numerical columns
sns.pairplot(loan_old[numerical_features])
plt.show()

#############################################

# c) Preprocess the data

# i) Remove records containing missing values
loan_old = loan_old.dropna()

# ii) the features and targets are separated
features = loan_old.drop(['Max_Loan_Amount', 'Loan_Status'], axis=1)
targets = loan_old[['Max_Loan_Amount', 'Loan_Status']]

# iii) Shuffle and split into training and testing sets
features_train, features_test, targets_train, targets_test = train_test_split(
    features, targets, test_size=0.3, random_state=33)

# iv) Categorical features encoding
cat_features = features.select_dtypes(include=['object']).columns
features[cat_features] = features[cat_features].apply(LabelEncoder().fit_transform)
features_train[cat_features] = features_train[cat_features].apply(LabelEncoder().fit_transform)
features_test[cat_features] = features_test[cat_features].apply(LabelEncoder().fit_transform)

# v) Categorical targets encoding
label_encoder = LabelEncoder()
targets['Loan_Status'] = label_encoder.fit_transform(targets['Loan_Status'])
targets_train['Loan_Status'] = label_encoder.transform(targets_train['Loan_Status'])
targets_test['Loan_Status'] = label_encoder.transform(targets_test['Loan_Status'])

# vi) Numerical features standardization
num_features = features.select_dtypes(include=['int64', 'float64']).columns
scaler = MinMaxScaler(feature_range=(-1, 1))
features[num_features] = scaler.fit_transform(features[num_features])
features_train[num_features] = scaler.transform(features_train[num_features])
features_test[num_features] = scaler.transform(features_test[num_features])

# Display the preprocessed data
print("Preprocessed Features:\n", features.head())
print("\nPreprocessed Targets:\n", targets.head())

##########################################


# d) Fit a linear regression model
linear_model = LinearRegression()
linear_model.fit(features_train, targets_train['Max_Loan_Amount'])

# e) Evaluate the linear regression model
linear_predictions = linear_model.predict(features_test)
linear_r2 = r2_score(targets_test['Max_Loan_Amount'], linear_predictions)
print("\nLinear Regression R2 Score:", linear_r2)

# f) Fit a logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(features_train, targets_train['Loan_Status'])

# g) Evaluate the logistic regression model
logistic_predictions = logistic_model.predict(features_test)
logistic_accuracy = accuracy_score(targets_test['Loan_Status'], logistic_predictions)
print("\nLogistic Regression Accuracy:", logistic_accuracy)

# h) Load the "loan_new.csv" dataset.
loan_new = pd.read_csv("loan_new.csv")

# i) Perform the same preprocessing on it (except shuffling and splitting).

# Remove records containing missing values
loan_new = loan_new.dropna()

# iv) Categorical features encoding
cat_loanNew = loan_new.select_dtypes(include=['object']).columns
loan_new[cat_loanNew] = loan_new[cat_loanNew].apply(LabelEncoder().fit_transform)

new_features = loan_new.select_dtypes(include=['int64', 'float64']).columns
scaler = MinMaxScaler(feature_range=(-1, 1))
loan_new[new_features] = scaler.fit_transform(loan_new[new_features])

newLinear_predictions = linear_model.predict(loan_new)
print("\nMax_Loan_Amount:", newLinear_predictions)

newLogistic_predictions = logistic_model.predict(loan_new)
print("\nLoan_Status:", newLogistic_predictions)

loan_new['Max_Loan_Amount'] = newLinear_predictions

loan_new['Loan_Status'] = newLogistic_predictions
