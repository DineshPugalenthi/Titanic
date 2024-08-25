#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Load the dataset (assuming 'train.csv' is your dataset file)
df = pd.read_csv("C:/Users/DELL/Desktop/ExcelR DS/Extracted Assignment files/Logistic Regression/Titanic_train.csv")

# Preprocessing
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Features and target variable
features = df[['Pclass', 'Sex', 'Age']]
target = df['Survived']

# Split the data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=100)

# Create a pipeline with a scaler and logistic regression
pipeline = make_pipeline(StandardScaler(), LogisticRegression())

# Train the model
pipeline.fit(x_train, y_train)

# Streamlit app
st.title('Titanic Survival Prediction')
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=0, max_value=100, value=30)

# Encode and predict
sex_encoded = le.transform([sex])[0]
input_data = pd.DataFrame([[pclass, sex_encoded, age]], columns=['Pclass', 'Sex', 'Age'])

# Prediction
prediction = pipeline.predict(input_data)
st.write(f"Survived: {'Yes' if prediction[0] == 1 else 'No'}")

