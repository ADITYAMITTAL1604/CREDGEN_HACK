import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from rapidfuzz import fuzz

# Load the LOF model
try:
    # Go up one level from 'agents' to root, then into 'models'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "models", "lof_pipeline.pkl")
    pipeline = joblib.load(model_path)
    MODEL_LOADED = True
except:
    pipeline = None
    MODEL_LOADED = False
    print("Warning: LOF model not found. Using rule-based only.")

class FraudAgent:
    def __init__(self):
        """Initialize Fraud Agent with ML model."""
        self.pipeline = pipeline
        self.model_loaded = MODEL_LOADED
    
    def perform_fraud_check(self, entities: dict) -> dict:
        """
        Main method called by app.py - performs comprehensive fraud check.
        Returns dict with fraud_score, fraud_flag, etc.
        """
        try:
            # Prepare customer data in format expected by predict_fraud
            cust_data = {
                'name_list': [entities.get('name', '')],
                'dob': entities.get('dob', ''),
                'age': entities.get('age', 35),
                'address': entities.get('address', ''),
                'salary': float(entities.get('income', 0) or 0),
                'emi_to_income_ratio': float(entities.get('emi_ratio', 0) or 0),
                'debt_to_income_ratio': float(entities.get('debt_ratio', 0) or 0),
                'active_loans': int(entities.get('existing_loans', 0) or 0),
                'requested_loan_amount': float(entities.get('loan_amount', 0) or 0)
            }
            
            # Call the existing predict_fraud function
            ml_result = predict_fraud(cust_data)
            
            # Run rule-based checks
            rule_result = self._rule_based_checks(entities)
            
            # Combine results
            fraud_score = (ml_result.get('anomaly_score', 0) * 0.7 + 
                          rule_result.get('rule_score', 0) * 0.3)
            
            # Determine fraud flag
            if fraud_score > 2:
                fraud_flag = 'High'
            elif fraud_score > 1.5:
                fraud_flag = 'Medium'
            else:
                fraud_flag = 'Low'
            
            return {
                'fraud_score': round(fraud_score, 3),
                'fraud_flag': fraud_flag,
                'ml_result': ml_result,
                'rule_based_result': rule_result,
                'entities_checked': list(entities.keys())
            }
            
        except Exception as e:
            print(f"Error in perform_fraud_check: {e}")
            return {
                'fraud_score': 0.5,
                'fraud_flag': 'Medium',
                'error': str(e)
            }
    
    def _rule_based_checks(self, entities: dict) -> dict:
        """Perform rule-based fraud checks."""
        score = 0
        flags = []
        
        # Check 1: Name consistency
        if 'name' in entities:
            name_score_data = name_score([entities['name']])
            if name_score_data['flag'] == 'HIGH':
                score += 0.3
                flags.append('name_mismatch')
        
        # Check 2: Income validation
        income = entities.get('income', 0)
        if income <= 0 or income > 50000000:
            score += 0.2
            flags.append('suspicious_income')
        
        # Check 3: Age validation
        if 'dob' in entities:
            age = dob_to_age(entities['dob'])
            if age and (age < 18 or age > 80):
                score += 0.2
                flags.append('invalid_age')
        
        # Check 4: Loan-to-income ratio
        income = entities.get('income', 1)
        loan_amount = entities.get('loan_amount', 0)
        if income > 0 and loan_amount / income > 20:
            score += 0.3
            flags.append('high_loan_to_income')
        
        return {
            'rule_score': min(score, 1.0),
            'flags': flags,
            'total_flags': len(flags)
        }

def name_score(name_list: list):
    name_list_cleaned = [name.strip().lower() for name in name_list if name and name.strip()]

    if len(name_list_cleaned)<2:
        return {"name_score": 1.0, "flag": "LOW"}
    score_ind = []
    for i in range(len(name_list_cleaned)):
        for j in range(i+1, len(name_list_cleaned)):
            name_1, name_2 = name_list_cleaned[i], name_list_cleaned[j]
            score = fuzz.token_set_ratio(name_1, name_2) /100
            score_ind.append(score)

    if min(score_ind)<0.8:
        flag = 'HIGH'
    else:
        flag = 'LOW'
    score_cum = sum(score_ind)/len(score_ind) 
    return {'name_score': score_cum, 'flag': flag}

def dob_to_age(dob_string):
    if not isinstance(dob_string, str) or dob_string.strip() == "":
        return np.nan

    for fmt in ("%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            dob = datetime.strptime(dob_string, fmt)
            today = datetime.today()
            return (today - dob).days / 365
        except:
            pass

    return np.nan

def extract_state_from_address(address):
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
        'punjab': ['punjab','chandigarh']
    }
    
    for state, keywords in state_keywords.items():
        for keyword in keywords:
            if keyword in address_lower:
                return state
    
    return "Other"

def predict_fraud(cust_details: dict):
    cust_details = cust_details.copy()

    cust_details['age'] = dob_to_age(cust_details['dob'])
    cust_details['state'] = extract_state_from_address(cust_details['address'])
    cust_details['name_in_application'] = cust_details['name_list'][0]
    cust_details['name_score'] = name_score(cust_details['name_list'])['name_score']
    cust_details['loan_to_salary_ratio'] = cust_details['requested_loan_amount'] / (cust_details['salary'] + 1)
    cust_details['total_debt_burden'] = cust_details['emi_to_income_ratio'] + cust_details['debt_to_income_ratio']
    cust_details['financial_stress'] = cust_details['debt_to_income_ratio'] * np.log1p(cust_details['active_loans'])

    features = [
        'state', 'name_score', 'age', 'salary', 'emi_to_income_ratio',
        'debt_to_income_ratio', 'active_loans', 'requested_loan_amount',
        'financial_stress', 'loan_to_salary_ratio', 'total_debt_burden'
    ]

    X = pd.DataFrame([[cust_details[f] for f in features]], columns=features)

    if pipeline and MODEL_LOADED:
        pred = pipeline.predict(X)
        raw_scores = pipeline.score_samples(X)
        anomaly = -raw_scores
        
        return {
            'fraud_flag': int(pred[0] == -1),
            'anomaly_score': float(anomaly[0]),
            'model_used': 'LOF'
        }
    else:
        # Fallback to rule-based
        return {
            'fraud_flag': 0,
            'anomaly_score': 0.3,
            'model_used': 'rule_based'
        }

# Test
'''if __name__ == "__main__":
    dummy_data = {
        'name_list': ['Rohit Sharma', 'Rohit K Sharma', 'Sharma Rohit'],
        'dob': '12-22-1995',
        'address': 'Delhi',
        'salary': 85000,
        'emi_to_income_ratio': 0.28,
        'debt_to_income_ratio': 0.35,
        'active_loans': 2,
        'requested_loan_amount': 450000
    }
    
    agent = FraudAgent()
    result = agent.perform_fraud_check({
        'name': 'Rohit Sharma',
        'dob': '12-22-1995',
        'address': 'Delhi',
        'income': 85000,
        'emi_ratio': 0.28,
        'debt_ratio': 0.35,
        'existing_loans': 2,
        'loan_amount': 450000
    })
    print(result)'''
