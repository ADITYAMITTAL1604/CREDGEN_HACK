import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Feature Engineering Function
def create_features(df):
    """Create engineered features"""
    df = df.copy()
    
    # Financial ratios
    df['income_to_loan_ratio'] = df['annual_income'] / (df['requested_loan_amount'] + 1)
    df['emi_affordability'] = df['monthly_income'] / (df['estimated_monthly_emi'] + 1)
    df['asset_coverage'] = df['property_value'] / (df['requested_loan_amount'] + 1)
    
    # Risk flags
    df['high_risk_flag'] = (
        (df['payment_history_default'] > 2) |
        (df['num_delinquent_accounts'] > 0) |
        (df['emi_to_income_ratio'] > 50) |
        (df['debt_to_income_ratio'] > 40)
    ).astype(int)
    
    # Stability score
    df['stability_score'] = df['years_employed'] * 0.3 + df['credit_age_months'] * 0.7
    
    return df

# Load and prepare data
print("="*60)
print("LOAN APPROVAL MODEL TRAINING")
print("="*60)

# Determine paths
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data", "loan_history.csv")
model_save_path = os.path.join(current_dir, "underwriting_model.pkl")

if not os.path.exists(data_path):
    print(f"Error: Data file not found at {data_path}")
    exit(1)

df = pd.read_csv(data_path)
df = create_features(df)

# Define features
numerical_features = [
    'age', 'years_employed', 'annual_income', 'monthly_income',
    'existing_loan_balance', 'existing_emi_monthly', 'credit_score', 
    'cibil_score', 'payment_history_default', 'credit_inquiry_last_6m',
    'num_open_accounts', 'num_delinquent_accounts', 'property_value',
    'requested_loan_amount', 'requested_loan_tenure', 'pre_approved_limit',
    'monthly_income_after_emi', 'debt_to_income_ratio', 'loan_to_income_ratio',
    'estimated_monthly_emi', 'emi_to_income_ratio', 'total_monthly_obligation',
    'obligation_to_income_ratio', 'loan_to_asset_ratio', 'credit_age_months',
    'income_to_loan_ratio', 'emi_affordability', 'asset_coverage',
    'stability_score'
]

categorical_features = [
    'gender', 'city', 'employment_type', 'education_level',
    'marital_status', 'home_ownership', 'property_type'
]

# Target
target = 'approval_status'
df['target'] = df[target].map({'Approved': 1, 'Rejected': 0})

print(f"\nDataset Information:")
print(f"Total samples: {len(df)}")
print(f"Approved: {df['target'].sum()} ({df['target'].mean():.2%})")
print(f"Rejected: {(df['target'] == 0).sum()} ({1 - df['target'].mean():.2%})")

# Train-test split
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

X = df[numerical_features + categorical_features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ============================================
# MODEL 1: LOGISTIC REGRESSION (Baseline)
# ============================================
print("\n" + "="*60)
print("TRAINING LOGISTIC REGRESSION")
print("="*60)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score

# Preprocessing for Logistic Regression
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_features),
        ('cat', cat_transformer, categorical_features)
    ]
)

# Logistic Regression model
logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        C=0.1, 
        penalty='l2', 
        solver='liblinear',
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ))
])

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
logreg_scores = cross_val_score(logreg_pipeline, X_train, y_train, 
                                 cv=cv, scoring='f1', n_jobs=-1)

# Train and evaluate
logreg_pipeline.fit(X_train, y_train)
y_pred_logreg = logreg_pipeline.predict(X_test)
y_pred_proba_logreg = logreg_pipeline.predict_proba(X_test)[:, 1]

f1_logreg = f1_score(y_test, y_pred_logreg)

print(f"Cross-validation F1: {logreg_scores.mean():.4f} (±{logreg_scores.std():.4f})")
print(f"Test F1 Score: {f1_logreg:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_logreg, 
                            target_names=['Rejected', 'Approved']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logreg))

# ============================================
# SAVE THE BEST MODEL
# ============================================
import joblib
import json

# Determine best model
# if f1_logreg > f1_catboost:
best_model = logreg_pipeline
best_model_name = "Logistic Regression"
best_f1 = f1_logreg
# else:
#     best_model = catboost_model
#     best_model_name = "CatBoost"
#     best_f1 = f1_catboost

# Save model
# Save model
joblib.dump(best_model, model_save_path)
best_threshold = 0.5

# Save feature names
model_info = {
    'model_name': best_model_name,
    'f1_score': float(best_f1),
    'features': {
        'numerical': numerical_features,
        'categorical': categorical_features,
        'engineered': ['income_to_loan_ratio', 'emi_affordability', 
                      'asset_coverage', 'high_risk_flag', 'stability_score']
    },
    'threshold': float(best_threshold),
    'dataset_size': len(df),
    'class_balance': {
        'approved': int(y.sum()),
        'rejected': int(len(y) - y.sum())
    }
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("\n" + "="*60)
print(f"BEST MODEL: {best_model_name}")
print(f"FINAL F1 SCORE: {best_f1:.4f}")
print("="*60)
print("Model saved as 'underwriting_model.pkl'")
print("Model info saved as 'model_info.json'")


# ============================================
# PREDICTION FUNCTION FOR NEW DATA
# ============================================
def predict_loan_approval(customer_data):
    """
    Predict loan approval for new customer data
    
    Parameters:
    customer_data: DataFrame with same columns as training data
    
    Returns:
    Dictionary with prediction and probabilities
    """
    # Load model
    # Load model
    model = joblib.load(model_save_path)
    
    # Add engineered features
    customer_data = create_features(customer_data)
    
    # Prepare features
    X_new = customer_data[numerical_features + categorical_features]
    
    # Predict
    probabilities = model.predict_proba(X_new)[:, 1]
    predictions = (probabilities >= best_threshold).astype(int)
    
    # Prepare results
    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        results.append({
            'customer_id': customer_data.iloc[i]['customer_id'] if 'customer_id' in customer_data.columns else i+1,
            'approval_status': 'Approved' if pred == 1 else 'Rejected',
            'approval_probability': float(prob),
            'confidence': 'High' if prob > 0.7 or prob < 0.3 else 'Medium',
            'recommendation': 'Approve' if pred == 1 else 'Reject'
        })
    
    return results

# ============================================
# VALIDATION ON TRAINING DATA
# ============================================
print("\n" + "="*60)
print("VALIDATION SUMMARY")
print("="*60)

# Training set performance
y_train_pred = best_model.predict(X_train)
y_train_proba = best_model.predict_proba(X_train)[:, 1]

train_f1 = f1_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_proba)

print(f"Training F1 Score: {train_f1:.4f}")
print(f"Training ROC-AUC: {train_auc:.4f}")
print(f"Test F1 Score: {best_f1:.4f}")

# Check for overfitting
if train_f1 - best_f1 > 0.15:
    print("\nModel may be overfitting!")
    print(f"   Difference: {train_f1 - best_f1:.4f}")
elif train_f1 - best_f1 > 0.10:
    print("\nModerate difference between train and test performance")
    print(f"   Difference: {train_f1 - best_f1:.4f}")
else:
    print(f"\n✓ Good generalization (difference: {train_f1 - best_f1:.4f})")

# ============================================
# NEXT STEPS RECOMMENDATION
# ============================================
print("\n" + "="*60)
print("NEXT STEPS RECOMMENDATION")
print("="*60)

if best_f1 < 0.85:
    print("Current F1 score is below target (< 0.85). Recommended actions:")
    print("1. Try XGBoost or LightGBM models")
    print("2. Add more feature engineering")
    print("3. Collect more training data if possible")
    print("4. Try ensemble methods")
elif best_f1 < 0.90:
    print("Good performance! To improve further:")
    print("1. Try hyperparameter tuning")
    print("2. Add more sophisticated feature engineering")
    print("3. Try stacking ensemble")
else:
    print("Excellent performance! Consider:")
    print("1. Deploying the model as-is")
    print("2. Adding explainability with SHAP values")
    print("3. Creating an API for predictions")

print(f"\nCurrent best model: {best_model_name}")
print(f"Current F1: {best_f1:.4f}")
print("="*60)

# Example usage for prediction
print("\nExample: Making predictions for new data")
print("-"*40)

# Create example new customer
example_data = {
    'customer_id': ['NEW001'],
    'age': [35],
    'gender': ['M'],
    'city': ['Bangalore'],
    'employment_type': ['Salaried'],
    'years_employed': [8],
    'annual_income': [800000],
    'monthly_income': [66666],
    'existing_loan_balance': [200000],
    'existing_emi_monthly': [5000],
    'credit_score': [750],
    'cibil_score': [800],
    'payment_history_default': [1],
    'credit_inquiry_last_6m': [2],
    'num_open_accounts': [4],
    'num_delinquent_accounts': [0],
    'education_level': ['Graduate'],
    'marital_status': ['Married'],
    'dependents': [2],
    'home_ownership': ['Mortgaged'],
    'property_type': ['Apartment'],
    'property_value': [5000000],
    'requested_loan_amount': [2000000],
    'requested_loan_tenure': [36],
    'pre_approved_limit': [2500000],
    'monthly_income_after_emi': [61666],
    'debt_to_income_ratio': [7.5],
    'loan_to_income_ratio': [30.0],
    'estimated_monthly_emi': [66666],
    'emi_to_income_ratio': [100.0],
    'total_monthly_obligation': [71666],
    'obligation_to_income_ratio': [107.5],
    'loan_to_asset_ratio': [40.0],
    'credit_age_months': [84]
}

example_df = pd.DataFrame(example_data)

# Add required columns that might be missing
for col in X.columns:
    if col not in example_df.columns:
        example_df[col] = 0  # Fill with default values

# Make prediction
try:
    predictions = predict_loan_approval(example_df)
    print("\nPrediction Result:")
    for pred in predictions:
        for key, value in pred.items():
            print(f"  {key}: {value}")
except Exception as e:
    print(f"Prediction error: {e}")
    print("Make sure all required features are present in the input data")


