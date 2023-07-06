import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

model = pickle.load(open('random_forest_model.pkl', 'rb'))


def main():
    st.title("Air Quality Index Prediction")
    st.write("Enter the feature values to predict the AQI:")
    
    # Create input fields for features
    co = st.number_input("CO", value=0.5)
    no2 = st.number_input("NO2", value=20)
    o3 = st.number_input("O3", value=30)
    pm10 = st.number_input("PM10", value=50)
    pm25 = st.number_input("PM2.5", value=25)
    so2 = st.number_input("SO2", value=10)
    ts = st.number_input("TS", value=1600000000)
    
    # Create a feature vector from input values
    features = np.array([[co, no2, o3, pm10, pm25, so2, ts]])
    
    # Make predictions using the loaded model
    prediction = model.predict(features)
    
    # Display the predicted AQI
    st.write("Predicted AQI:", prediction[0])


if __name__ == "__main__":
    main()
