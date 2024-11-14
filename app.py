import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the pre-trained model
model = joblib.load("random_forest_model.pkl")

# Set up the title and description of the web app
st.title("Income Prediction App")
st.write("This app predicts whether a person's income is less than or greater than $50k based on their education, occupation, country, and weekly hours worked.")

# Define the input fields
full_name = st.text_input("Full Name")
education = st.selectbox("Education Level", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
country = st.selectbox("Country", ["United-States", "Canada", "Mexico", "Philippines", "Germany", "India", "Other"])
weekly_hours = st.number_input("Weekly Hours Worked", min_value=1, max_value=100, value=40)

# Map user inputs to the format expected by the model
# Ensure the mappings for 'education', 'occupation', and 'country' align with your training dataset encoding
education_mapping = {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "Doctorate": 4}
occupation_mapping = {"Tech-support": 0, "Craft-repair": 1, "Sales": 2, "Exec-managerial": 3, "Prof-specialty": 4, "Handlers-cleaners": 5, "Machine-op-inspct": 6, "Adm-clerical": 7, "Farming-fishing": 8, "Transport-moving": 9, "Priv-house-serv": 10, "Protective-serv": 11, "Armed-Forces": 12}
country_mapping = {"United-States": 0, "Canada": 1, "Mexico": 2, "Philippines": 3, "Germany": 4, "India": 5, "Other": 6}

# Convert inputs to model-ready format
education_encoded = education_mapping.get(education)
occupation_encoded = occupation_mapping.get(occupation)
country_encoded = country_mapping.get(country)
input_features = np.array([[education_encoded, occupation_encoded, country_encoded, weekly_hours]])

# Make prediction when user clicks "Predict"
if st.button("Predict"):
    prediction = model.predict(input_features)
    income_label = "Greater than 50k" if prediction[0] == 1 else "Less than 50k"
    
    st.write(f"Prediction for {full_name}: {income_label}")

# Footer
st.write("Note: This prediction is made based on a machine learning model.")
