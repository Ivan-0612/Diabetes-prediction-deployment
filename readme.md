# Diabetes Prediction System

An interactive web application built with Streamlit and Machine Learning that predicts the probability of a patient having diabetes based on demographic data, habits, and medical metrics.
Features

## Features

* Interactive Form: Simple data entry for the user (age, BMI, glucose, etc.).
* Real-Time Prediction: Uses a trained Logistic Regression model to calculate risk instantly.
* Explainability (XAI): Displays SHAP plots to explain which variables increased or decreased the specific patient's risk.

## Project Structure

```text
├── app.py                # Main Streamlit application code
├── deploy_model.pkl      # Trained model (Pipeline + Logistic Regression)
├── requirements.txt      # List of required libraries
└── README.md             # Project documentation

```

The project is deployed on Render: https://diabetes-prediction-rlji.onrender.com/
