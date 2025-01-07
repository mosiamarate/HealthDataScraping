
# HealthDatabase

# Project Number: 1
# Project Title: HealthDatabase
# Code Version: 1.0
# Written by: Mosia Marate
# Date: 202-09-30


"""
HealthDatabase Project

This project focuses on predicting diabetes in patients using the Pima Indians Diabetes dataset.
The steps include data loading, preprocessing, exploratory data analysis (EDA), model building,
evaluation, model interpretation, and making predictions on new data. Additionally, we save and
load the trained model for future use.

Libraries Used:
- numpy: For numerical computations
- pandas: For data manipulation and analysis
- matplotlib: For data visualization
- seaborn: For data visualization
- sklearn: For machine learning tasks, including model building and evaluation
- joblib: For saving and loading the model

Dataset:
The dataset used is the Pima Indians Diabetes dataset, which contains several medical predictor
variables and one target variable, Outcome. The predictor variables include:
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age (years)
- Outcome: Class variable (0 or 1)

Steps:
1. Data Loading: Load the dataset from a URL.
2. Data Preprocessing: Handle missing values by imputing with statistical measures.
3. Exploratory Data Analysis (EDA): Visualize data distribution and relationships.
4. Feature Engineering: Split data into features and target, and standardize the features.
5. Model Building: Train a Random Forest Classifier.
6. Model Evaluation: Evaluate the model using accuracy, confusion matrix, and classification report.
7. Model Interpretation: Interpret the model using feature importance.
8. Model Saving and Loading: Save and load the trained model.
9. Making Predictions: Make predictions on new data.

"""

# Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=column_names)

# Real-world Scenario: Data Cleaning and Preprocessing
# In real-world datasets, missing values are common. We handle these by replacing 0s with NaNs and then imputing with statistical measures.
data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
data['Glucose'].fillna(data['Glucose'].mean(), inplace=True)
data['BloodPressure'].fillna(data['BloodPressure'].mean(), inplace=True)
data['SkinThickness'].fillna(data['SkinThickness'].median(), inplace=True)
data['Insulin'].fillna(data['Insulin'].median(), inplace=True)
data['BMI'].fillna(data['BMI'].mean(), inplace=True)

# Exploratory Data Analysis (EDA)
# Visualization helps us understand data distribution and relationships
plt.figure(figsize=(15, 10))
data.hist(bins=15, figsize=(15, 10))
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Split the data into features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Real-world Scenario: Feature Engineering
# Standardizing features is crucial for models like Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Building and Evaluation
# In practice, Random Forest is robust and interpretable
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# Model Interpretation
# Feature importance can offer insights into which factors are most influential
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title("Feature Importance")
plt.show()

# Save the model
joblib.dump(model, 'diabetes_model.pkl')
print("Model Saved Successfully")

# Load the model
model = joblib.load('diabetes_model.pkl')
print("Model Loaded Successfully")

# Real-world Scenario: Making Predictions
# Here we simulate making a prediction for a new patient
new_patient_data = {
    'Pregnancies': 5,
    'Glucose': 120,
    'BloodPressure': 70,
    'SkinThickness': 30,
    'Insulin': 100,
    'BMI': 25,
    'DiabetesPedigreeFunction': 0.5,
    'Age': 30
}

new_patient_df = pd.DataFrame([new_patient_data])
new_patient_df = scaler.transform(new_patient_df)
prediction = model.predict(new_patient_df)

if prediction[0]:
    print("The Patient has Diabetes")
else:
    print("The Patient does not have Diabetes")

# Additional Real-world Considerations:
# 1. Model Maintenance: Regularly retrain your model with new data to keep it up-to-date.
# 2. Ethical Considerations: Ensure data privacy and obtain patient consent for using medical data.
# 3. Interpretability: Consider using models like Random Forest that offer interpretability to understand and trust predictions.
# 4. Deployment: Deploy the model in a user-friendly interface, such as a web app, for healthcare professionals to use in practice.

# End of HealthDatabase.py
