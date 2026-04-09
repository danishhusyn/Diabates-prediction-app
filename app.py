import streamlit as st
import pickle
import numpy as np

import os

base_path = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(base_path, 'model/model.pkl'), 'rb'))
scaler_model = pickle.load(open(os.path.join(base_path, 'model/scaler.pkl'), 'rb'))

# App title
st.title("Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0)
insulin = st.number_input("Insulin Level", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0, step=1)

# Predict button
if st.button("Predict"):
    # Prepare input
    features = np.array([[pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, bmi, dpf, age]])
    
    features_scaled = scaler_model.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1] * 100

    # Risk classification
    if probability >= 70:
        risk = "High Risk"
        st.error(" High Risk of Diabetes")
    elif probability >= 40:
        risk = "Moderate Risk"
        st.warning(" Moderate Risk of Diabetes")
    else:
        risk = "Low Risk"
        st.success(" Low Risk of Diabetes")

    result = "Diabetic" if prediction == 1 else "Non-Diabetic"

    # Display results
    st.subheader("Prediction Result")
    st.write(f"**Result:** {result}")
    st.write(f"**Risk Level:** {risk}")
    st.write(f"**Probability:** {probability:.2f}%")
