## Libraries Importation 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# Import and Defining the dataframe
path = r"C:\Users\user\Desktop\JW\Jewelry-Price-Optimization-main\Jewelry-Price-Optimization\raw_data.csv"
df = pd.read_csv(path)

# Display columns to identify mismatch
print(df.columns)

# Update the column names to match the data
column_name = ['Order ID', 'Purchased product ID', 'Quantity of SKU in the order', 'Category ID', 'Category alias', 
               'Brand ID', 'Price in USD', 'User ID', 'Product gender (for male/female)', 'Main Color', 'Main metal', 
               'Main gem', 'Unnamed']  

# Renaming the DataFrame columns
df.columns = column_name
df.head()

# Data exploration
df.info()

# Dropping duplicate entries to clean the dataset
df.drop_duplicates(inplace=True)

# Handling missing values in the target variable 'Price in USD' by filling with the mean
df['Price in USD'].fillna(df['Price in USD'].mean(), inplace=True)

# Displaying a statistical summary of the dataset
print(df.describe())

# Function to split the data into features (X) and target (y)
def split(data):
    x = data.drop(['Price in USD'], axis=1)
    y = data['Price in USD']
    return x, y

# Splitting the dataset
x, y = split(df)

# Function to process the data using pipelines for numerical and categorical features
def process(data):
    # Pipeline for numerical features: Imputation and Scaling
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline for categorical features: Imputation and Encoding
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # Combining both pipelines using ColumnTransformer
    transformer = ColumnTransformer(
        transformers=[
            ('num_features', num_pipe, data.select_dtypes(include=['int', 'float']).columns),
            ('cat_features', cat_pipe, data.select_dtypes(include=['object']).columns)
        ],
        remainder='drop'
    )
    
    # Transforming the data
    return transformer.fit_transform(data)

# Processing the features
x_processed = process(x)

# Function to split the processed data into training and testing sets
def train(a, b):
    x_train, x_test, y_train, y_test = train_test_split(a, b, test_size=0.25, random_state=42)
    return x_train, x_test, y_train, y_test

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train(x_processed, y)

# Function to train and evaluate multiple regression models
def models():
    # Dictionary to store various regression models
    models = {
        'LR': LinearRegression(),
        'RF': RandomForestRegressor(),
        'XGB': XGBRegressor(),
        'TREE': DecisionTreeRegressor(),
        'GR': GradientBoostingRegressor(),
        'KN': KNeighborsRegressor()
    }

    # Print model performance evaluation
    print('Model Performance Evaluation')
    for name, model in models.items():
        # Fit the model and make predictions
        prediction = model.fit(x_train, y_train).predict(x_test)
        # Print evaluation metrics for each model
        print(f"The Coefficient of Determination (R2 Score) for {name} model is {r2_score(y_test, prediction)}")
        print(f"The Mean Absolute Error (MAE) for {name} model is {mean_absolute_error(y_test, prediction)}")
        print(f"The Mean Absolute Percentage Error (MAPE) for {name} model is {mean_absolute_percentage_error(y_test, prediction)}")

# Evaluate the models
algorithm = models()
