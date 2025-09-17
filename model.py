import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Load the dataset
data = pd.read_csv('heart_disease.csv')

# Preprocess data (basic preprocessing for simplicity)
X = data.drop(columns='target')  # Features
y = data['target']               # Target variable

# Check if the target variable has both classes
print("Unique values in target before split:", np.unique(y))

# Add synthetic data if only one class is present in the target
if len(np.unique(y)) == 1:
    print("Only one class found in target. Adding synthetic data for testing.")
    
    # Number of synthetic samples to add (adjust as needed)
    num_samples = 5
    
    # Create synthetic rows with the other class (if existing class is 1, add class 0)
    new_rows = pd.DataFrame({col: X.iloc[0][col] for col in X.columns}, index=range(num_samples))
    new_rows['target'] = 0  # Assign a different class value

    # Append synthetic rows to the dataset
    data = pd.concat([data, new_rows], ignore_index=True)

    # Re-define X and y after synthetic data is added
    X = data.drop(columns='target')
    y = data['target']

# Split data into train and test sets with stratification to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ✅ Save feature names to avoid mismatch error
feature_names = X_train.columns

# Check for class distribution in y_train after splitting
print("Unique values in y_train:", np.unique(y_train))
print("Training set class distribution:", np.bincount(y_train))
print("Testing set class distribution:", np.bincount(y_test))

# Initialize the XGBoost Classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the model
model.fit(X_train, y_train)

# ✅ Reassign feature names to match the training data before prediction
X_test.columns = feature_names

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# ✅ Save the trained model and feature names
joblib.dump((model, feature_names), 'heart_disease_model.pkl')
print("Model saved as heart_disease_model.pkl")
