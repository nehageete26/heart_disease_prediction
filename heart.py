import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("heart (1).csv")
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

target = df['target'].value_counts()
print(target)

# 1: defective heart 2: healthy heart 
X = df.drop(columns='target',axis=1)
y = df['target']

# training and testing model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model training 
model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
score = accuracy_score(y_test,y_pred)

st.title("Heart Disease Prediction")
st.write(f"Model trained with accuracy: *{score:.2f}*")

# User input using streamlit -> 
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"][x])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], format_func=lambda x: [
"Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"][x])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=210, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of the peak exercise ST segment", options=[0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3) colored by fluoroscopy", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], format_func=lambda x: ["Null", "Normal", "Fixed defect", "Reversible defect"][x])

# input -> 
input_data = np.array([[age, sex, cp, trestbps, chol, fbs,restecg, thalach, exang, oldpeak, slope, ca, thal]])

if st.button("Predict disease"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("High risk of heart disease detected.")
    else:
        st.success("No heart disease detected.")
