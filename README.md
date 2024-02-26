# House Price Prediction
This project aims to predict house prices using machine learning models. I used the Ridge Regression model for prediction, as it gives the best results in terms of RMSE and MAPE.

##  Overview
### Data Collection: 
First I'm start by reading the dataset from a CSV file using pandas.

### Data Preprocessing & Exploratory Data Analysis (EDA):
I'm perform various preprocessing steps, such as dropping irrelevant columns ('ID'), handling missing values (if any), checking for outliers, and splitting the data into training and testing sets.
#### Steps for Data Preprocessing
(i) Drop Irrelevant Columns: I'm droping the 'ID' column as it is not relevant for model training.

(ii) Handle Missing Values: I'm checking for and handle any missing values in the dataset.

(iii) Check for Outliers: Using box plots to identify and handle any outliers in the data.

(iv) Split the Data: I split the data into training and testing sets using a 80-20 split ratio.

(V)  I also performed to visualize the data using a correlation heatmap and box plots to understand the relationships between variables and identify outliers.

### Model Selection: 
I compare several regression models, including Linear Regression, Ridge Regression, Lasso Regression, Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor. Finally I choose the Ridge Regression model based on its performance metrics.

### Model Training: 
I trained the Ridge Regression model on the training data.

### Model Evaluation: 
Evaluate the model using metrics such as RMSE, MSE, MAE, and MAPE on the test data.

Select the Best Model: Selected the Ridge Regression model based on its performance metrics.

### Saving the Model: 
Finally save the trained Ridge Regression model using pickle for future use.

## Assumptions
I assumed that the dataset is representative of the housing market and does not contain significant biases.
I assumed that the features used in the model are sufficient to predict house prices accurately.

## Conclusion
The Ridge Regression model performs well in predicting house prices based on the given dataset. Additional features or data sources could potentially improve the model's performance.

## "House Price Prediction Output"


![Image Alt text](/kanerika_Assign_results.gif)
