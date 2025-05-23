# -*- coding: utf-8 -*-
"""2_trip_hist_anal.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sxEJgBnTyWUxxXzalU_CNYt-ECptWb31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("sample_trip_history.csv")

# Convert Start Time to datetime
df['Start Time'] = pd.to_datetime(df['Start Time'])

# Exploratory Data Analysis (EDA)
print("Dataset Head:\n", df.head())
print("\nDataset Info:\n")
df.info()
print("\nSummary Statistics:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# Visualizations
plt.figure(figsize=(8, 5))
sns.countplot(x='User Type', data=df)
plt.title("User Type Distribution")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Trip Duration'], bins=30, kde=True)
plt.title("Trip Duration Distribution")
plt.show()

# Feature Engineering
df['Hour'] = df['Start Time'].dt.hour
df['Day of Week'] = df['Start Time'].dt.dayofweek
df.drop(columns=['Start Time', 'End Time'], inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['Start Station'] = le.fit_transform(df['Start Station'])
df['End Station'] = le.fit_transform(df['End Station'])
df['User Type'] = le.fit_transform(df['User Type'])  # Target variable

# Splitting dataset
X = df.drop(columns=['User Type'])
y = df['User Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

# Predictions
y_pred = dt.predict(X_test)

# Evaluation
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualizing the Decision Tree
plt.figure(figsize=(15, 8))
plot_tree(dt_model, feature_names=X.columns, class_names=['Casual', 'Subscriber'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
