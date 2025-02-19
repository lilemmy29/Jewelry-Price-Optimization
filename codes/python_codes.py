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
from sklearn.feature_selection import RFE


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




# Feature Selection and Engineering 

## Creating new features
df['Total Order Value'] = df['Quantity of SKU in the order'] * df['Price in USD']
df['Product Popularity'] = df.groupby('Purchased product ID')['Purchased product ID'].transform('count')
df['Category Popularity'] = df.groupby('Category ID')['Category ID'].transform('count')

# Function to split the data into features (X) and target (y)
def split(data):
    x = data.drop(['Price in USD'], axis=1)
    y = data['Price in USD']
    return x, y

# Splitting the dataset
x, y = split(df)

# Function to process the data using pipelines for numerical and categorical features
def updated_process(data):
    # Updating numerical features to include the new features
    numerical_features = data.select_dtypes(include=['int', 'float']).columns.tolist()
    numerical_features.extend(['Total Order Value', 'Product Popularity', 'Category Popularity'])
    
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
            ('num_features', num_pipe, numerical_features),
            ('cat_features', cat_pipe, data.select_dtypes(include=['object']).columns)
        ],
        remainder='drop'
    )
    
    # Transforming the data
    return transformer.fit_transform(data)

# Processing the features with the updated process
x_updated = updated_process(x)

# Function to split the processed data into training and testing sets
def train(a, b):
    x_train, x_test, y_train, y_test = train_test_split(a, b, test_size=0.25, random_state=42)
    return x_train, x_test, y_train, y_test

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train(x_updated, y)

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
updated_algorithm = models()


# Define function to perform feature selection and evaluate models
def feature_selection_and_evaluation(models, x_train, x_test, y_train, y_test):
    # Feature Selection and Engineering
    
    # 1. Correlation analysis
    correlation_matrix = df.corr()
    
    # 2. Feature importance using tree-based models (e.g., RandomForest)
    rf = RandomForestRegressor()
    rf.fit(x_train, y_train)
    feature_importances = rf.feature_importances_
    
    # 3. Recursive Feature Elimination (RFE) for Linear Regression
    lr = LinearRegression()
    rfe = RFE(estimator=lr, n_features_to_select=10, step=1)
    rfe.fit(x_train, y_train)
    selected_features_lr = x.columns[rfe.support_]
    
    # 4. Model-Based Feature Selection (e.g., coefficients for LR)
    lr.fit(x_train, y_train)
    coefficients_lr = lr.coef_
    selected_features_lr_coef = x.columns[np.abs(coefficients_lr) > threshold]
    
    # 5. Combine selected features based on analysis above
    
    # Model Evaluation
    for name, model in models.items():
        # Fit the model and make predictions
        prediction = model.fit(x_train[selected_features], y_train).predict(x_test[selected_features])
        
        # Print evaluation metrics for each model
        print(f"Model: {name}")
        print(f"Selected Features: {selected_features}")
        print(f"R2 Score: {r2_score(y_test, prediction)}")
        print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, prediction)}")
        print(f"Mean Absolute Percentage Error (MAPE): {mean_absolute_percentage_error(y_test, prediction)}")
        print("------------------------------------------------------")

# Models dictionary
models = {
    'LR': LinearRegression(),
    'RF': RandomForestRegressor(),
    'XGB': XGBRegressor(),
    'TREE': DecisionTreeRegressor(),
    'GR': GradientBoostingRegressor(),
    'KN': KNeighborsRegressor()
}

# Execute feature selection and evaluation
feature_selection_and_evaluation(models, x_train, x_test, y_train, y_test)




# Define hyperparameter distributions for each model
param_dist_lr = {
    'fit_intercept': [True, False]
}

param_dist_rf = {
    'n_estimators': randint(100, 300),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4)
}

param_dist_xgb = {
    'n_estimators': randint(100, 300),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 7),
    'subsample': uniform(0.6, 0.4)
}

param_dist_tree = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4)
}

param_dist_gr = {
    'n_estimators': randint(100, 300),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 7),
    'subsample': uniform(0.6, 0.4)
}

param_dist_kn = {
    'n_neighbors': randint(3, 7),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Initialize the models
models = {
    'LR': (LinearRegression(), param_dist_lr),
    'RF': (RandomForestRegressor(), param_dist_rf),
    'XGB': (XGBRegressor(), param_dist_xgb),
    'TREE': (DecisionTreeRegressor(), param_dist_tree),
    'GR': (GradientBoostingRegressor(), param_dist_gr),
    'KN': (KNeighborsRegressor(), param_dist_kn)
}

# Hyper-tuning function using RandomizedSearchCV
def hyper_tune(models, x_train, y_train):
    best_models = {}
    for name, (model, param_dist) in models.items():
        random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50, cv=3, scoring='r2', n_jobs=-1, verbose=2, random_state=42)
        random_search.fit(x_train, y_train)
        best_models[name] = random_search.best_estimator_
        print(f"Best parameters for {name}: {random_search.best_params_}")
    return best_models

# Hyper-tune the models
best_models = hyper_tune(models, x_train, y_train)

# Evaluate the best models
print('Best Model Performance Evaluation')
for name, model in best_models.items():
    prediction = model.predict(x_test)
    print(f"The Coefficient of Determination (R2 Score) for {name} model is {r2_score(y_test, prediction)}")
    print(f"The Mean Absolute Error (MAE) for {name} model is {mean_absolute_error(y_test, prediction)}")
    print(f"The Mean Absolute Percentage Error (MAPE) for {name} model is {mean_absolute_percentage_error(y_test, prediction)}")
