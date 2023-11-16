<h1>Loan Regression ML</h1>
This repository contains code for a machine learning project that predicts loan amounts and loan statuses based on a dataset. The project involves data analysis, preprocessing,
and the training of linear regression and logistic regression models.


<h2>Dataset</h2>
The project uses two datasets:

loan_old.csv: This dataset is used for analysis, preprocessing, and training the initial models.
loan_new.csv: This dataset is used to demonstrate the model's predictions after training.

Dataset Files

Colab Notebook: <a href="https://colab.research.google.com/drive/13xqokz7tlLfVi2qvRo-BqNPZhQMMCrtT?usp=sharing" target="_blank">Colab Notebook</a>
 
For a detailed walkthrough of the project, including code and visualizations, refer to the Colab notebook: Colab Notebook



python main_script.py
Analysis and Preprocessing
Analysis: The dataset is analyzed for missing values, data types, and numerical feature scales. A pairplot is visualized to explore relationships between numerical columns.

Preprocessing: The data is preprocessed by removing records with missing values, separating features and targets, shuffling and splitting the data, encoding categorical features and targets,
and standardizing numerical features.

Linear Regression Model
Model: A linear regression model is fitted to predict the maximum loan amount.

Evaluation: The model is evaluated using the R2 score on the test set.

Logistic Regression Model
Model: A logistic regression model is fitted to predict loan approval status.

Evaluation: The model is evaluated using accuracy on the test set.

Applying the Models to New Data
loan_new.csv: The preprocessing steps are applied to the new dataset, and both linear and logistic regression models are used to predict loan amounts and statuses, respectively.
Results
The predictions on the new dataset are stored in the loan_new.csv file, including columns for predicted Max_Loan_Amount and Loan_Status.
