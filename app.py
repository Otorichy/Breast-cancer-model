import streamlit as st
import pickle
import numpy as np

# Load model

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Logistic Regression Prediction App")

st.write("Enter the feature values below:")

inputs = []

for i in range(len(model.coef_[0])):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(value)

if st.button("Predict"):
    X = np.array(inputs).reshape(1, -1)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    st.success(f"Prediction: {prediction}")
    st.info(f"Probability of class 1: {probability:.2f}")

