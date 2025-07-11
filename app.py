import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load saved model, scaler, and feature means
model = joblib.load('logistic_model.pkl')  # or 'ridge_model.pkl' if using ridge_reg
scaler = joblib.load('scaler.pkl')
feature_means = joblib.load('feature_means.pkl')
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Streamlit app
st.title("Diabetes Probability Prediction")
st.write("Enter your details to predict your diabetes probability.")

# Input form
age = st.number_input("Age [0-100]", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
glucose = st.number_input("Glucose [50-300]", min_value=50.0, max_value=300.0, value=100.0, step=1.0)
blood_pressure = st.number_input("BloodPressure [50-200]", min_value=50.0, max_value=200.0, value=70.0, step=1.0)

if st.button("Predict"):
    # Create input vector
    input_data = np.array([[feature_means['Pregnancies'], glucose, blood_pressure, feature_means['SkinThickness'], 
                            feature_means['Insulin'], feature_means['BMI'], feature_means['DiabetesPedigreeFunction'], age]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Overall probability
    overall_prob = model.predict_proba(input_scaled)[0, 1] * 100

    # Individual probabilities
    input_age = input_data.copy()
    input_age[:, features.index('Glucose')] = feature_means['Glucose']
    input_age[:, features.index('BloodPressure')] = feature_means['BloodPressure']
    age_prob = model.predict_proba(scaler.transform(input_age))[0, 1] * 100

    input_glucose = input_data.copy()
    input_glucose[:, features.index('Age')] = feature_means['Age']
    input_glucose[:, features.index('BloodPressure')] = feature_means['BloodPressure']
    glucose_prob = model.predict_proba(scaler.transform(input_glucose))[0, 1] * 100

    input_bp = input_data.copy()
    input_bp[:, features.index('Age')] = feature_means['Age']
    input_bp[:, features.index('Glucose')] = feature_means['Glucose']
    bp_prob = model.predict_proba(scaler.transform(input_bp))[0, 1] * 100

    # Display probabilities
    st.write(f"For Age: {age_prob:.2f}% probability of diabetes")
    st.write(f"For Glucose: {glucose_prob:.2f}% probability of diabetes")
    st.write(f"For BloodPressure: {bp_prob:.2f}% probability of diabetes")
    st.write(f"Overall chances of being diabetic: {overall_prob:.2f}%")

    # Pie charts
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    labels = ['Diabetic', 'Non-Diabetic']
    colors = ['#ff9999', '#66b3ff']

    axs[0, 0].pie([age_prob, 100 - age_prob], labels=labels, autopct='%1.1f%%', colors=colors)
    axs[0, 0].set_title('Age Probability')

    axs[0, 1].pie([glucose_prob, 100 - glucose_prob], labels=labels, autopct='%1.1f%%', colors=colors)
    axs[0, 1].set_title('Glucose Probability')

    axs[1, 0].pie([bp_prob, 100 - bp_prob], labels=labels, autopct='%1.1f%%', colors=colors)
    axs[1, 0].set_title('BloodPressure Probability')

    axs[1, 1].pie([overall_prob, 100 - overall_prob], labels=labels, autopct='%1.1f%%', colors=colors)
    axs[1, 1].set_title('Overall Probability')

    plt.tight_layout()
    st.pyplot(fig)