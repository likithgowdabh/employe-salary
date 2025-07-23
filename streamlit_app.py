# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("salary_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("Employee Salary Prediction")

experience = st.slider("Years of Experience", 0, 30)
education = st.selectbox("Education", ["Bachelors", "Masters", "PhD"])
job_role = st.selectbox("Job Role", ["Developer", "Manager", "Analyst"])
industry = st.selectbox("Industry", ["IT", "Finance", "Healthcare"])
location = st.selectbox("Location", ["Bangalore", "Mumbai", "Delhi"])

input_data = {
    "Experience": experience,
    "Education": label_encoders["Education"].transform([education])[0],
    "JobRole": label_encoders["JobRole"].transform([job_role])[0],
    "Industry": label_encoders["Industry"].transform([industry])[0],
    "Location": label_encoders["Location"].transform([location])[0]
}

input_df = pd.DataFrame([input_data])

if st.button("Predict Salary"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Salary: ₹{int(prediction):,}")
