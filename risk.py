import streamlit as st
import pandas as pd
import pickle

dataset = pd.read_csv("dataset.csv")

with open("stroke_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Stroke Risk Prediction")

gender = st.selectbox("Gender", sorted(dataset["gender"].unique()))
age = st.number_input("Age", min_value=0, max_value=120)
hypertension = st.selectbox("Hypertension", sorted(dataset["hypertension"].unique()))
heart_disease = st.selectbox("Heart Disease", sorted(dataset["heart_disease"].unique()))
ever_married = st.selectbox("Ever Married", sorted(dataset["ever_married"].unique()))
work_type = st.selectbox("Work Type", sorted(dataset["work_type"].unique()))
Residence_type = st.selectbox("Residence Type", sorted(dataset["Residence_type"].unique()))
avg_glucose_level = st.number_input("Average Glucose Level")
smoking_status = st.selectbox("Smoking Status", sorted(dataset["smoking_status"].unique()))

if st.button("Predict"):
    data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": Residence_type,
        "avg_glucose_level": avg_glucose_level,
        "smoking_status": smoking_status
    }

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    if prediction == 1:
        st.error("Stroke Risk")
    else:
        st.success("No Stroke Risk")



