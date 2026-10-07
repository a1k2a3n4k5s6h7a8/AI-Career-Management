# train_model.py
# This script trains a machine learning model to predict student placement status
# and another model to estimate the salary package (LPA) for placed students.
# It preprocesses the data, compares multiple algorithms, and saves the best model.

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import the classification models we want to compare
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Import the regression models for salary estimation
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def train_and_evaluate():
    # -------------------------------------------------------------
    # STEP 1: LOAD THE DATASET
    # -------------------------------------------------------------
    print("Loading the dataset...")
    # Define path to the dataset
    dataset_path = os.path.join("dataset", "student_placement_prediction_dataset_2026.csv")
    
    # Read the CSV file using pandas
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please check the directory structure.")
        
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.\n")

    # -------------------------------------------------------------
    # STEP 2: PREPARE FEATURES (X) AND TARGETS (Y)
    # -------------------------------------------------------------
    print("Preparing features and target variables...")
    
    # Target variable for classification (placement prediction)
    # Map 'Placed' to 1 and 'Not Placed' to 0
    df['placed_binary'] = df['placement_status'].map({'Placed': 1, 'Not Placed': 0})
    
    # Drop columns that are not features:
    # - 'student_id' is just an incremental ID (no predictive value)
    # - 'placement_status' and 'placed_binary' are the target variables
    # - 'salary_package_lpa' must be removed to prevent data leakage (since it is 0 for unplaced, >0 for placed)
    features_to_drop = ['student_id', 'placement_status', 'placed_binary', 'salary_package_lpa']
    X = df.drop(columns=features_to_drop)
    y = df['placed_binary']
    
    # -------------------------------------------------------------
    # STEP 3: IDENTIFY COLUMN TYPES & DEFINE PREPROCESSING PIPELINE
    # -------------------------------------------------------------
    # Categorize features into numerical and categorical lists
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}\n")
    
    # Define preprocessing for numerical columns:
    # 1. Impute missing values with mean (just in case there are any)
    # 2. Scale features to have mean=0 and variance=1 (StandardScaler)
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical columns:
    # 1. Impute missing values with the most frequent value
    # 2. One-hot encode the categorical variables (handle unknown categories by ignoring them)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine both transformers into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # -------------------------------------------------------------
    # STEP 4: TRAIN-TEST SPLIT
    # -------------------------------------------------------------
    print("Splitting data into training and test sets (80% train, 20% test)...")
    # Split the raw features and target. The preprocessor will be fit inside our pipeline.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]} rows")
    print(f"Testing set size: {X_test.shape[0]} rows\n")

    # -------------------------------------------------------------
    # STEP 5: COMPARE CLASSIFICATION MODELS
    # -------------------------------------------------------------
    # Define a dictionary of models to train and compare
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    best_model_name = None
    best_model_pipeline = None
    best_accuracy = 0.0
    model_comparison_results = []
    
    print("--- Training and Comparing Classification Models ---")
    for name, model in models.items():
        print(f"Training {name}...")
        # Create a pipeline that combines the preprocessor and the classifier
        clf_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train the model pipeline
        clf_pipeline.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = clf_pipeline.predict(X_test)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Results for {name}:")
        print(f"  Accuracy  : {accuracy:.4f}")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1-Score  : {f1:.4f}\n")
        
        model_comparison_results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        # Save the model pipeline if it's the best one based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model_pipeline = clf_pipeline

    print("--- Summary Comparison Table ---")
    comparison_df = pd.DataFrame(model_comparison_results)
    print(comparison_df.to_string(index=False))
    print(f"\nBest Model selected: {best_model_name} with Accuracy: {best_accuracy:.4f}\n")

    # Ensure model directory exists
    os.makedirs("model", exist_ok=True)

    # Save the best classification pipeline (includes preprocessor + model)
    classification_model_path = os.path.join("model", "placement_classifier.joblib")
    joblib.dump(best_model_pipeline, classification_model_path)
    print(f"Saved the best classification model to {classification_model_path}")

    # -------------------------------------------------------------
    # STEP 6: TRAIN SALARY REGRESSOR FOR PLACED STUDENTS
    # -------------------------------------------------------------
    print("\n--- Training Salary Regressor (LPA) for Placed Students ---")
    # Filter the dataset to include only placed students for the regression model
    placed_df = df[df['placement_status'] == 'Placed'].copy()
    
    # X_reg contains the same input features, but only for placed students
    X_reg = placed_df.drop(columns=features_to_drop)
    y_reg = placed_df['salary_package_lpa']
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    print(f"Salary Regressor - Training size: {X_train_reg.shape[0]} rows")
    print(f"Salary Regressor - Testing size: {X_test_reg.shape[0]} rows")
    
    # We will use a Random Forest Regressor for robust package estimation
    regressor = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    
    # Create a pipeline combining the preprocessor and the regressor
    regressor_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    # Train the regressor pipeline
    print("Training Random Forest Regressor...")
    regressor_pipeline.fit(X_train_reg, y_train_reg)
    
    # Predict on regression test set
    y_pred_reg = regressor_pipeline.predict(X_test_reg)
    
    # Calculate performance metrics
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    
    print(f"Salary Regressor Evaluation:")
    print(f"  Mean Absolute Error (MAE): {mae:.4f} LPA")
    print(f"  R-squared (R2 Score)     : {r2:.4f}\n")
    
    # Save the regression pipeline (includes preprocessor + regressor model)
    regression_model_path = os.path.join("model", "salary_regressor.joblib")
    joblib.dump(regressor_pipeline, regression_model_path)
    print(f"Saved the salary regressor model to {regression_model_path}")
    
    # -------------------------------------------------------------
    # STEP 7: SAVE FEATURE INFO FOR DYNAMIC FORM RENDERING
    # -------------------------------------------------------------
    # Save the list of feature column names and classes to help app.py validate inputs
    feature_info = {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'categorical_options': {col: df[col].unique().tolist() for col in categorical_cols}
    }
    feature_info_path = os.path.join("model", "feature_info.joblib")
    joblib.dump(feature_info, feature_info_path)
    print(f"Saved feature info metadata to {feature_info_path}\n")
    
    print("All models and preprocessors trained and saved successfully!")

if __name__ == "__main__":
    train_and_evaluate()
