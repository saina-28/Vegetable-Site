import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import streamlit as st

# Load the dataset
data = pd.read_csv('vegetable_prices.csv')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'], format='mixed')

# Sort the data by date
data = data.sort_values('Date')

# Set 'Date' as the index
data.set_index('Date', inplace=True)

# Streamlit app title with custom background color
st.title("Estimation of Vegetable Prices")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #228B22;
        
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# User input for the vegetable to predict
vegetable_to_predict = st.text_input("Vegetable Name:")

# Define custom CSS style for the button
button_style = """
    <style>
    div.stButton > button {
        background-color: #228B22;
        color: #d9edc4;
        display: block;
        margin: 0 auto;
    }
    </style>
    """

# Render the custom CSS style
st.markdown(button_style, unsafe_allow_html=True)

# Button to trigger prediction
if st.button("Predict Price"):
    # Filter data for the specific vegetable
    vegetable_data = data[data['Item Name'] == vegetable_to_predict]

    if vegetable_data.empty:
        st.error("Vegetable not found in the dataset.")
    else:
        # Select the target column 'price'
        target_column = 'price'
        target_data = vegetable_data[[target_column]]

        # Normalize the target data
        scaler = MinMaxScaler()
        target_scaled = scaler.fit_transform(target_data)

        # Define the sequence length
        sequence_length = 10

        # Prepare the data for LSTM
        X = []
        y = []

        for i in range(sequence_length, len(target_scaled)):
            X.append(target_scaled[i - sequence_length:i, 0])
            y.append(target_scaled[i, 0])

        X = np.array(X)
        y = np.array(y)

        # Reshape the input data for LSTM (samples, time steps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Build the LSTM model (same as in your code)
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
            LSTM(units=50),
            Dense(units=1)
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model (same as in your code)
        model.fit(X, y, epochs=50, batch_size=32)

        # Predict today's vegetable price
        last_sequence = target_scaled[-sequence_length:].reshape(1, sequence_length, 1)
        predicted_price_scaled = model.predict(last_sequence)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)

        st.success(f"Predicted Price for Today ({vegetable_to_predict}): {predicted_price[0][0]}")



