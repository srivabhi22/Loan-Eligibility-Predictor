import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

# Load the saved model
model = joblib.load('model.pkl')

# Set up the page title
st.title("Loan Eligibility Predictor")

# Collect user input
st.header("Please enter your details below:")

loan_id = st.text_input('Loan ID')
gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
married = st.selectbox('Married', ['Yes', 'No'])
dependents = st.text_input('Dependents')
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
applicant_income = st.number_input('Applicant Income', min_value=0, step=1)
coapplicant_income = st.number_input('Coapplicant Income', min_value=0.0, step=0.01, format="%.2f")
loan_amount = st.number_input('Loan Amount', min_value=0.0, step=0.01, format="%.2f")
loan_amount_term = st.number_input('Loan Amount Term', min_value=0.0, step=0.01, format="%.2f")
credit_history = st.selectbox('Credit History', [1.0, 0.0])
property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])
loan_status = st.selectbox('Loan Status', ['Y', 'N'])

# Input validation
if st.button('Predict'):
    if not loan_id or not gender or not married or not education or not property_area:
        st.error("Please fill in all required fields.")
    elif not isinstance(applicant_income, int) or not isinstance(coapplicant_income, float):
        st.error("Applicant Income must be an integer and Coapplicant Income must be a float.")
    elif not isinstance(credit_history, float) or not isinstance(loan_amount_term, float) or not isinstance(loan_amount, float):
        st.error("Credit History, Loan Amount Term, and Loan Amount must be floats.")
    else:
        # Create a DataFrame from user input
        user_input = pd.DataFrame({
            'Married': [married],
            'Education': [education],
            'CoapplicantIncome': [coapplicant_income],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        })

        # Encoding
        ord_encoder = OrdinalEncoder()
        user_input[['Married', 'Education']] = ord_encoder.fit_transform(user_input[['Married', 'Education']])
        
        ohe = OneHotEncoder(drop='first', sparse=False)
        encoded_property_area = ohe.fit_transform(user_input[['Property_Area']])
        encoded_df = pd.DataFrame(encoded_property_area, columns=ohe.get_feature_names_out(['Property_Area']))
        user_input = pd.concat([user_input.drop('Property_Area', axis=1), encoded_df], axis=1)

        # Scaling
        scaler = StandardScaler()
        scaled_input = scaler.fit_transform(user_input)

        # Prediction
        prediction = model.predict(scaled_input)

        # Output the result
        if prediction == 1:
            st.success('Congratulations! You are eligible for the loan.')
        else:
            st.warning('Sorry, you are not eligible for the loan.')
