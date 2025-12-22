import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer

def dob_to_age(dob_string):
    """Convert DOB to age."""
    if pd.isna(dob_string) or not isinstance(dob_string, str) or dob_string.strip() == "":
        return np.nan
    
    dob_string = dob_string.strip()
    
    date_formats = ["%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y"]
    
    for fmt in date_formats:
        try:
            dob = datetime.strptime(dob_string, fmt)
            today = datetime.today()
            age = (today - dob).days / 365.25
            if 18 <= age <= 100:
                return age
        except:
            continue
    
    return np.nan

def extract_state_from_address(address):
    """Extract state from address."""
    if pd.isna(address):
        return "Unknown"
    
    address_lower = str(address).lower()
    
    state_keywords = {
        'tamil nadu': ['tamil nadu', 'chennai'],
        'maharashtra': ['maharashtra', 'mumbai', 'pune'],
        'delhi': ['delhi'],
        'karnataka': ['karnataka', 'bengaluru'],
        'uttar pradesh': ['uttar pradesh', 'lucknow'],
        'west bengal': ['west bengal', 'kolkata'],
        'gujarat': ['gujarat', 'ahmedabad'],
        'rajasthan': ['rajasthan', 'jaipur'],
        'telangana': ['telangana', 'hyderabad'],
        'assam': ['assam', 'guwahati'],
        'madhyapradesh': ['madhya pradesh', 'bhopal'],
        'kerala': ['kerala', 'thiruvananthapuram'],
        'bihar': ['bihar', 'patna'],
        'punjab': ['punjab', 'chandigarh']
    }
    
    for state, keywords in state_keywords.items():
        for keyword in keywords:
            if keyword in address_lower:
                return state
    
    return "Other"

def main():
    # Load data
    df = pd.read_csv("kyc_sample.csv")
    
    # Create features
    df['age'] = df['dob'].apply(dob_to_age)
    df['state'] = df['address'].apply(extract_state_from_address)
    
    # Engineered features for better anomaly detection
    df['loan_to_salary_ratio'] = df['requested_loan_amount'] / (df['salary'] + 1)
    df['total_debt_burden'] = df['emi_to_income_ratio'] + df['debt_to_income_ratio']
    df['financial_stress'] = df['debt_to_income_ratio'] * np.log1p(df['active_loans'])
    
    # Define features
    # categorical_features = ['state']
    categorical_features = ['state']
    # numerical_features = [
    #     'name_score', 'age', 'salary', 'emi_to_income_ratio',
    #     'debt_to_income_ratio', 'active_loans', 'requested_loan_amount',
    #     'loan_to_salary_ratio', 'total_debt_burden', 'financial_stress'
    # ]
    numerical_features = [
        'name_score', 'age', 'salary', 'emi_to_income_ratio',
        'debt_to_income_ratio', 'active_loans', 'requested_loan_amount',
        'financial_stress'
    ]
    
    # Prepare data
    X = df[categorical_features + numerical_features]
    
    # Build pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features),
            
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numerical_features),
        ]
    )
    
    # LOF model
    lof = LocalOutlierFactor(
        n_neighbors=25,
        contamination=0.01,
        novelty=True
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("lof", lof)
    ])
    
    # Train
    pipeline.fit(X)
    
    # Save ONLY the pipeline
    joblib.dump(pipeline, "lof_pipeline.pkl")

if __name__ == "__main__":
    main()

