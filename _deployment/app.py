import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle


city="random_forest"
model_file = city.lower() + "_model.pkl"
model = pickle.load(open(model_file, 'rb'))

def main():
    st.title("Air Quality Index Prediction")
    
    city = st.selectbox("Select City", ["Ahmedabad", "Bengaluru", "Chandigarh", "Chennai",  "Delhi"])
    model_file = city.lower() + "_model.pkl"
    model = pickle.load(open(model_file, 'rb'))
    
    st.write("Enter the feature values to predict the AQI:")
    co = st.number_input("CO", value=0.5)
    no2 = st.number_input("NO2", value=20)
    o3 = st.number_input("O3", value=30)
    pm10 = st.number_input("PM10", value=50)
    pm25 = st.number_input("PM2.5", value=25)
    so2 = st.number_input("SO2", value=10)
    ts = st.number_input("TS", value=1600000000)
    
    features = np.array([[co, no2, o3, pm10, pm25, so2, ts]])
    prediction = model.predict(features)
    
    st.write("Predicted AQI:", prediction[0])



if __name__ == "__main__":
    main()
