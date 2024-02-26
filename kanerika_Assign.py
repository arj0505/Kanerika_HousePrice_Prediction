
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pickle

# Load the model
model_path = r'C:\Users\arj05\OneDrive\Desktop\Assignment\house_price.pickle'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

st.title('House Price Prediction App 🏠')

# Upload CSV data
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    #Data preprocessing
    df.drop('ID', axis=1, inplace=True)
    X = df.drop(['SalePrice'], axis=1)
    Y = df[['SalePrice']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Save the model
    st.write('Saving the trained model...')
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    # Prediction
    st.subheader('House Price Prediction')
    st.write('Enter values for prediction:')
    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(col, value=X[col].mean())
    input_df = pd.DataFrame([input_data])

    if st.button('Predict'):
        prediction = model.predict(input_df)
        #st.write(f"Predicted SalePrice: {prediction[0]:.2f}")
        st.write(f"Predicted SalePrice: {prediction[0]}")

    
