## Importing neccesary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

## Import and Defining the dataframe
df = pd.read_csv('Jewelry_Dataset.csv')
df

## Appending the variables names into the dataframe
column_name = ['Order datetime', 'Order ID', 'Purchased product ID', 'Quantity of SKU in the order', 'Category ID', 'Category alias', 'Brand ID', 'Price in USD', 'User ID', 'Product gender (for male/female)', 'Main Color', 'Main metal', 'Main gem']
df.columns = column_name
df.head()

#data exploration
df.info()

#dropping duplicate
df.drop_duplicates(inplace=True)

## Statistical Summary
df.describe()


## Splitting the dataset into Dependent(y) and Independent Varaibles(x) 
x = df.drop(['Price in USD'],axis=1)
y = df['Price in USD']


## Splitting Variables(x,y) into training and testing sets  
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


def processor(data):
    # Make a copy of the input data to avoid modifying the original
    data = data.copy()

    # Identify numerical and categorical columns
    num_cols = data.select_dtypes(include=["int", "float"]).columns
    cat_cols = data.select_dtypes(include=["object"]).columns

    # Numerical pipeline: Impute missing values with mean and scale numerical columns
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Replace missing values in numerical columns with mean
        ('scaler', StandardScaler())  # Standardize numerical columns for better model performance
    ])

    # Categorical pipeline: Impute missing values with 'unknown' and encode categorical columns
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),  # Replace missing values in categorical columns with 'unknown'
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))  #Encode categorical columns to numerical values
    ])

    # Combine numerical and categorical pipelines into a ColumnTransformer
    transformer = ColumnTransformer(
        transformers=[
            ("numeric_pipe", num_pipe, num_cols),  # Apply num_pipe to numerical columns
            ("categorical_pipe", cat_pipe, cat_cols)  # Apply cat_pipe to categorical columns
        ],
        remainder="passthrough",  # Pass through any columns not specified in num_cols or        cat_cols
        n_jobs=-1  # Use all available CPUs for parallel processing
    )



    return transformer

processed_data = processor(df)

def get_model(data):
    # List of model tuples with names and instantiated model objects
    models = [
        ("randomforest_model", RandomForestRegressor()),
        ("logistic_model", LogisticRegression()),
        ("decisiontree_model", DecisionTreeRegressor()),
        ("knn_model", KNeighborsRegressor()),
        ("svm_model", SVR()),
        ("xgboost_model", XGBRegressor())
    ]

    # Create a list to store model pipelines
    model_pipelines = []

    # Loop through each model and create a pipeline with the preprocessor
    for name, model in models:
        # Create a pipeline that applies the preprocessor to the data and then the model
        model_pipeline = make_pipeline(processed_data, model)
        # Append the tuple of model name and its pipeline to the list
        model_pipelines.append((name, model_pipeline))

    # Return the list of model pipelines
    return model_pipelines
get_model(df)

## Model Evaluation 

'''
def evaluate_model(pipeline, x_train, y_train, x_test, y_test):
    # Fit the model
    pipeline.fit(x_train, y_train)
    # Predict on the test set
    y_pred = pipeline.predict(x_test)
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

# Evaluate all models
evaluation_results = {name: evaluate_model(pipeline, x_train, y_train, x_test, y_test) for name, pipeline in model_pipelines}
evaluation_results
'''

