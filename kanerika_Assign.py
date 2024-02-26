## Below code is used Streamlit for Streamlit cloud deployment
#Importing the necessary Python libraries.
import pandas as pd
import streamlit as st
import pickle
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")


import requests
import pickle

# URL of the pickled model
model_url = 'https://github.com/arj0505/Kanerika_HousePrice_Prediction/raw/main/house_price.pickle'

# Download the model
response = requests.get(model_url)
with open('house_price.pickle', 'wb') as f:
    f.write(response.content)

# Load the model
with open('house_price.pickle', 'rb') as model_file:
    model = pickle.load(model_file)


# Load the model
#model_path = 'https://github.com/arj0505/Kanerika_HousePrice_Prediction/blob/main/house_price.pickle'
#with open(model_path, 'rb') as model_file:
#    model = pickle.load(model_file)

st.title('House Price Prediction App 🏠')

# Prediction
st.subheader('House Price Prediction')
st.write('Enter values for prediction:')
X=['OverallQual','GrLivArea','YearBuilt','TotalBsmtSF','FullBath','HalfBath','GarageCars','GarageArea']
input_data = {}
for col in X:
    input_data[col] = st.number_input(col, value=0)  # Set a default value of 0
input_df = pd.DataFrame([input_data])

if st.button('Predict'):
    prediction = model.predict(input_df)
    st.write(f"Predicted SalePrice: {prediction[0]}")




