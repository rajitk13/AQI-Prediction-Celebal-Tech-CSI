import streamlit as st
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

# Load the data


# Define the Streamlit app
def main():
    st.title("Air Quality Index Prediction")
    city = st.selectbox("Select City", ["Ahmedabad", "Bengaluru", "Chandigarh", "Chennai",  "Delhi"])
    
    data = pd.read_csv("../citywise-one-month-datasets/one_month_{}.csv".format(city))
    df = data.drop(['co', 'timestamp_local', 'timestamp_utc', 'no2', 'o3', 'pm10', 'pm25', 'so2', 'ts'], axis='columns')

    # Prepare the data for training
    Y = df['aqi']
    Y = np.array(Y)
    Y = Y.reshape(-1,1)
    L = len(df)
    X1 = Y[0:L-5,:]
    X2 = Y[1:L-4,:]
    X3 = Y[2:L-3,:]
    X4 = Y[3:L-2,:]
    X5 = Y[4:L-1,:]
    Y = Y[5:L,:]
    X = np.concatenate([X1,X2,X3,X4,X5],axis=1)

    # Scale the data
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    scaler1 = MinMaxScaler()
    scaler1.fit(Y)
    Y = scaler1.transform(Y)
    X = np.reshape(X, (X.shape[0],1,X.shape[1]))

    X_train = X[:500,:,:]
    X_test = X[500:,:,:]
    Y_train = Y[:500,:]
    Y_test = Y[500:,:]

    # Train the LSTM model
    model = Sequential()
    model.add(LSTM(10,activation = 'tanh',input_shape = (1,5),recurrent_activation= 'hard_sigmoid'))
    model.add(Dense(1))

    model.compile(loss= 'mean_squared_error',optimizer = 'rmsprop', metrics=[metrics.mae])
    model.fit(X_train,Y_train,epochs=20,verbose=2)
    Predict = model.predict(X_test)

    predictions = scaler1.inverse_transform(Predict)
    Y_test = scaler1.inverse_transform(Y_test)

    forecasted_aqi = predictions[-1*24:]


    # Load the Random Forest model

    model_file = city.lower() + "_model.pkl"
    random_forest_model = pickle.load(open(model_file, 'rb'))
 
    model_file = city.lower() + "_model.pkl"
    random_forest_model = pickle.load(open(model_file, 'rb'))
    
    st.write("Enter the feature values to predict the AQI:")
    co = st.number_input("CO", value=0.5)
    no2 = st.number_input("NO2", value=20)
    o3 = st.number_input("O3", value=30)
    pm10 = st.number_input("PM10", value=50)
    pm25 = st.number_input("PM2.5", value=25)
    so2 = st.number_input("SO2", value=10)
    ts = st.number_input("TS", value=1600000000)
    
    features = np.array([[co, no2, o3, pm10, pm25, so2, ts]])
 
    random_forest_prediction = random_forest_model.predict(features)
    
 
    st.write("Random Forest Predicted AQI:", random_forest_prediction[0])

    st.write("Time Series AQI:",forecasted_aqi)

   

if __name__ == "__main__":
    main()
