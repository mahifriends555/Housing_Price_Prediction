# app.py - Streamlit app for California Housing Price Prediction
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved models and scaler
lr_model = joblib.load('lr_model.pkl')
xgb_best_model = joblib.load('xgb_best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define feature columns (hardcoded, including engineered features)
feature_columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedInc_AveRooms', 'Dist_SF']

# Streamlit app layout
st.title('California Housing Price Prediction')
st.write('Enter the feature values to predict the median house value using Linear Regression or XGBoost.')

# User inputs for original features
med_inc = st.number_input('Median Income (MedInc)', min_value=0.0, max_value=15.0, value=3.0)
house_age = st.number_input('House Age (HouseAge)', min_value=1.0, max_value=52.0, value=30.0)
ave_rooms = st.number_input('Average Rooms (AveRooms)', min_value=1.0, max_value=10.0, value=5.0)
ave_bedrms = st.number_input('Average Bedrooms (AveBedrms)', min_value=0.5, max_value=5.0, value=1.0)
population = st.number_input('Population', min_value=0.0, max_value=5000.0, value=1000.0)
ave_occup = st.number_input('Average Occupancy (AveOccup)', min_value=0.5, max_value=5.0, value=2.5)
latitude = st.number_input('Latitude', min_value=32.0, max_value=42.0, value=37.0)
longitude = st.number_input('Longitude', min_value=-125.0, max_value=-114.0, value=-122.0)


# Create DataFrame for input
input_data = pd.DataFrame({
    'MedInc': [med_inc],
    'HouseAge': [house_age],
    'AveRooms': [ave_rooms],
    'AveBedrms': [ave_bedrms],
    'Population': [population],
    'AveOccup': [ave_occup],
    'Latitude': [latitude],
    'Longitude': [longitude],
    # 'MedInc_AveRooms': [med_inc_ave_rooms],
    
    # 'Dist_SF': [dist_sf]
})

# Scale input data
input_scaled = scaler.transform(input_data)

# Model selection
model_choice = st.selectbox('Select Model', ['Linear Regression', 'XGBoost (Best Tuned)'])

# Predict button
if st.button('Predict'):
    if model_choice == 'Linear Regression':
        log_pred = lr_model.predict(input_scaled)[0]
    else:
        log_pred = xgb_best_model.predict(input_scaled)[0]
    
    # Transform back to original scale
    pred = np.expm1(log_pred) * 100000  # Multiply by 100,000 for actual value
    
    st.success(f'Predicted Median House Value: ${pred:,.2f}')