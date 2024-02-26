# House Price Prediction
This project aims to predict house prices using machine learning models. We use the Ridge Regression model for prediction, as it gives the best results in terms of RMSE and MAPE.

##  Overview
### Data Collection: 
We start by reading the dataset from a CSV file using pandas.

### Data Preprocessing & Exploratory Data Analysis (EDA):
We perform various preprocessing steps, such as dropping irrelevant columns ('ID'), handling missing values (if any), checking for outliers, and splitting the data into training and testing sets.
#### Steps for Data Preprocessing
(i) Drop Irrelevant Columns: We drop the 'ID' column as it is not relevant for model training.

(ii) Handle Missing Values: We check for and handle any missing values in the dataset.

(iii) Check for Outliers: We use box plots to identify and handle any outliers in the data.

(iv) Split the Data: We split the data into training and testing sets using a 80-20 split ratio.

(V)  We visualize the data using a correlation heatmap and box plots to understand the relationships between variables and identify outliers.

### Model Selection: 
We compare several regression models, including Linear Regression, Ridge Regression, Lasso Regression, Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor. We choose the Ridge Regression model based on its performance metrics.

### Model Training: 
We train the Ridge Regression model on the training data.

### Model Evaluation: 
We evaluate the model using metrics such as RMSE, MSE, MAE, and MAPE on the test data.

Select the Best Model: We select the Ridge Regression model based on its performance metrics.

### Saving the Model: 
We save the trained Ridge Regression model using pickle for future use.

## Assumptions
We assume that the dataset is representative of the housing market and does not contain significant biases.
We assume that the features used in the model are sufficient to predict house prices accurately.

## Conclusion
The Ridge Regression model performs well in predicting house prices based on the given dataset. Additional features or data sources could potentially improve the model's performance.

## "House Price Prediction Output"


![Image Alt text](/kanerika_Assign_results.gif)
