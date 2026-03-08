"""
Corporate Profit Prediction Pipeline
This script loads corporate spending data, performs Exploratory Data Analysis (EDA), 
engineers financial ratio features, and trains multiple regression models (Linear, 
Random Forest, XGBoost, LightGBM) utilizing RandomizedSearchCV to find and save 
the best performing model.
"""

import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA LOADING & EXPLORATION
# ==========================================
FILE_PATH = 'dataframe.csv'

try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Error: Dataset '{FILE_PATH}' not found. Please ensure it is in the same directory.")
    sys.exit(1)

print("--- Dataframe Head ---\n", df.head())
print("\n--- Null Count ---\n", df.isnull().sum())
print("\n--- Data Information ---")
df.info()

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
# Initial numeric columns for correlation analysis
initial_num_cols = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']

# Generate Correlation Heatmap
print("\nGenerating Correlation Heatmap. (Note: Close the plot window to continue execution...)")
plt.figure(figsize=(8, 6))
sns.heatmap(df[initial_num_cols].corr(), annot=True, fmt='.2g', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Generate Histograms
print("Generating Feature Histograms. (Note: Close the plot window to continue execution...)")
df[initial_num_cols].hist(bins=15, figsize=(12, 6))
plt.suptitle('Feature Distributions')
plt.show()


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
# Add a small epsilon (1e-5) to prevent division by zero
df['RD_to_Admin_ratio'] = df['R&D Spend'] / (df['Administration'] + 1e-5)
df['Marketing_to_Admin_ratio'] = df['Marketing Spend'] / (df['Administration'] + 1e-5)
df['Total_Spend'] = df['R&D Spend'] + df['Administration'] + df['Marketing Spend']


# ==========================================
# 4. PREPROCESSING & DATA SPLITTING
# ==========================================
X = df.drop('Profit', axis=1)
y = df['Profit']

# Dynamically define categorical and numerical columns based on updated feature set
cat_col = ['State']
num_col =[col for col in X.columns if col not in cat_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define column transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_col),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_col)
    ]
)


# ==========================================
# 5. MODEL CONFIGURATION
# ==========================================
pipelines = {
    'Linear Regression': Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())]),
    'Random Forest': Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42))]),
    'XGBoost': Pipeline([('preprocessor', preprocessor), ('model', XGBRegressor(random_state=42))]),
    'LightGBM': Pipeline([('preprocessor', preprocessor), ('model', LGBMRegressor(random_state=42, verbose=-1))])
}

# Define safe, model-specific hyperparameter grids to prevent algorithmic crashes
param_grids = {
    'Random Forest': {
        "model__n_estimators":[100, 200, 300, 500],
        "model__max_depth":[3, 5, 7, 10, None],
        "model__min_samples_split":[2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4]
    },
    'XGBoost': {
        "model__n_estimators":[100, 200, 300, 500],
        "model__max_depth":[3, 5, 7, 10, None]
    },
    'LightGBM': {
        "model__n_estimators":[100, 200, 300, 500],
        "model__max_depth":[3, 5, 7, 10, None]
    }
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)


# ==========================================
# 6. MODEL TRAINING & TUNING
# ==========================================
best_mae = float('inf')
best_model_name = ""
best_model_pipeline = None

print("\n--- Starting Model Training & Hyperparameter Tuning ---")

for name, pipeline in pipelines.items():
    
    if name == 'Linear Regression':
        # Linear Regression doesn't need hyperparameter tuning
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        current_estimator = pipeline
        
    else:
        # Perform RandomizedSearchCV for Tree-based models
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grids[name],
            n_iter=50,
            scoring='neg_mean_absolute_error',
            cv=kf,
            random_state=42,
            n_jobs=-1  # Utilize all CPU cores for faster tuning
        )
        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        current_estimator = search.best_estimator_
        
    print(f"Model: {name}")
    print(f"Test MAE: {mae:.4f}\n") 

    # Track the best performing model
    if mae < best_mae:
        best_mae = mae
        best_model_name = name
        best_model_pipeline = current_estimator
  
print('='*45)
print(f"🏆 The Winning Model: {best_model_name}")
print(f"🏆 Best MAE: {best_mae:.2f}")
print('='*45,'\n')


# ==========================================
# 7. EXPORT MODEL
# ==========================================
model_filename = 'best_model.joblib'
joblib.dump(best_model_pipeline, model_filename)
print(f"Success! The best model pipeline has been saved as '{model_filename}'")