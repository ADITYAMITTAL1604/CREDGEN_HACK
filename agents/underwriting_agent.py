import numpy as np
import pickle
import joblib
import os
import pandas as pd

# --- AI MODEL TRAINING AND LOADING ---
def load_underwriting_model(filepath: str):
    """Attempts to load the trained model from disk."""
    if os.path.exists(filepath):
        print(f"Loading REAL AI Model from {filepath}...")
        try:
            return joblib.load(filepath)
        except Exception as e:
            print(f"Failed to load REAL model: {e}. Falling back to MOCK.")
            return None
    return None

# --- MOCK AI MODEL (Fallback/Development) ---
class MockModel:
    def predict_proba(self, data):
        """Simulates the model output (risk score) using actual column names."""
        income = data['annual_income'].iloc[0] if 'annual_income' in data.columns else 500000
        cibil = data['cibil_score'].iloc[0] if 'cibil_score' in data.columns else 700
        loan_amount = data['requested_loan_amount'].iloc[0] if 'requested_loan_amount' in data.columns else 500000
        age = data['age'].iloc[0] if 'age' in data.columns else 35
        
        print(f"\n=== MockModel Debug ===")
        print(f"  Income: ₹{income:,}")
        print(f"  CIBIL: {cibil}")
        print(f"  Loan Amount: ₹{loan_amount:,}")
        print(f"  Age: {age}")
        
        # Heuristic: Start with moderate risk
        risk = 0.50
        
        # Income factor (higher income = lower risk)
        if income >= 2000000:
            risk -= 0.25
            print(f"  Income ≥ 20L: risk -= 0.25")
        elif income >= 1000000:
            risk -= 0.15
            print(f"  Income ≥ 10L: risk -= 0.15")
        elif income >= 500000:
            risk -= 0.05
            print(f"  Income ≥ 5L: risk -= 0.05")
        
        # CIBIL factor (higher score = lower risk)
        if cibil >= 750:
            risk -= 0.20
            print(f"  CIBIL ≥ 750: risk -= 0.20")
        elif cibil >= 700:
            risk -= 0.10
            print(f"  CIBIL ≥ 700: risk -= 0.10")
        elif cibil >= 650:
            risk -= 0.05
            print(f"  CIBIL ≥ 650: risk -= 0.05")
        elif cibil < 600:
            risk += 0.20
            print(f"  CIBIL < 600: risk += 0.20")
        
        # Loan-to-income ratio (higher ratio = higher risk)
        lti_ratio = loan_amount / income if income > 0 else 1.0
        print(f"  Loan-to-Income Ratio: {lti_ratio:.2f}")
        if lti_ratio > 5:
            risk += 0.30
            print(f"  LTI > 5: risk += 0.30")
        elif lti_ratio > 3:
            risk += 0.20
            print(f"  LTI > 3: risk += 0.20")
        elif lti_ratio > 2:
            risk += 0.10
            print(f"  LTI > 2: risk += 0.10")
        elif lti_ratio <= 0.5:
            risk -= 0.10
            print(f"  LTI ≤ 0.5 (excellent): risk -= 0.10")
        
        # Age factor
        if age < 25:
            risk += 0.05
            print(f"  Age < 25: risk += 0.05")
        elif age > 55:
            risk += 0.05
            print(f"  Age > 55: risk += 0.05")
        
        risk = max(0.05, min(0.95, risk))
        
        print(f"  FINAL RISK SCORE: {risk:.3f}")
        print(f"======================\n")
        
        # Return format: [P(no default), P(default)]
        return np.array([[1 - risk, risk]])


class UnderwritingAgent:
    def __init__(self):
        """Initializes agent and loads the AI predictive model."""
        # --- RULE-BASED LAYER: Hard Business Policy Constants ---
        self.MIN_AGE = 21
        self.MAX_AGE = 60
        self.MIN_INCOME = 300000
        self.MAX_LOAN = 2000000
        self.RISK_THRESHOLD_REJECT = 1.5  # AI Score > 1.5 is Auto-Reject Rule
        
        # Define feature lists
        self.numerical_features = [
            'age', 'years_employed', 'annual_income', 'monthly_income',
            'existing_loan_balance', 'existing_emi_monthly', 'credit_score',
            'cibil_score', 'payment_history_default', 'credit_inquiry_last_6m',
            'num_open_accounts', 'num_delinquent_accounts', 'property_value',
            'requested_loan_amount', 'requested_loan_tenure', 'pre_approved_limit',
            'monthly_income_after_emi', 'debt_to_income_ratio', 'loan_to_income_ratio',
            'estimated_monthly_emi', 'emi_to_income_ratio', 'total_monthly_obligation',
            'obligation_to_income_ratio', 'loan_to_asset_ratio', 'credit_age_months',
            'income_to_loan_ratio', 'emi_affordability', 'asset_coverage', 'stability_score'
        ]
        
        self.categorical_features = [
            'gender', 'city', 'employment_type', 'education_level',
            'marital_status', 'home_ownership', 'property_type'
        ]
        
        # Full model features order
        self.model_features_order = self.numerical_features + self.categorical_features
        
        # --- AI LAYER: Load Model ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "..", "models", "underwriting_model.pkl")
        
        loaded_model = load_underwriting_model(model_path)
        if loaded_model is None:
            print("Using MOCK AI Model for Underwriting.")
            self.model = MockModel()
        else:
            self.model = loaded_model
            print("Successfully loaded REAL AI Model.")
        
        print("✓ Underwriting Agent ready.\n")
    
    def _hard_reject(self, reason: str) -> dict:
        """Helper to format a standardized rejection response."""
        return {
            "approval_status": False,
            "risk_score": 1.0,
            "interest_rate": None,
            "reason": f"HARD REJECTED: {reason}"
        }
    
    def _mock_interest_rate(self, risk_score: float) -> float:
        """Simplified pricing rule based on the AI Risk Score."""
        BASE_RATE = 9.5
        MAX_RATE = 18.0
        rate = BASE_RATE + (risk_score * (MAX_RATE - BASE_RATE))
        return round(min(rate, MAX_RATE), 2)
    
    def _preprocess_input(self, entities: dict) -> pd.DataFrame:
        """
        Creates a DataFrame with proper feature engineering and categorical encoding.
        """
        print(f"\n=== Preprocessing Input ===")
        print(f"Raw entities received: {entities}")
        
        # Extract core values from entities
        age = int(entities.get('age', 35))
        annual_income = float(entities.get('annual_income', 500000))
        loan_amount = float(entities.get('requested_loan_amount', 500000))
        tenure = int(entities.get('requested_loan_tenure', 36))
        cibil = int(entities.get('cibil_score', 750))
        employment_type_raw = str(entities.get('employment_type', 'Salaried'))
        
        # Calculate derived values
        monthly_income = annual_income / 12
        estimated_emi = loan_amount / tenure  # Simplified EMI calculation
        
        # --- Numerical features with calculated values ---
        numerical_defaults = {
            'age': age,
            'years_employed': 5,
            'annual_income': annual_income,
            'monthly_income': monthly_income,
            'existing_loan_balance': 0,
            'existing_emi_monthly': 0,
            'credit_score': cibil,
            'cibil_score': cibil,
            'payment_history_default': 0,
            'credit_inquiry_last_6m': 1,
            'num_open_accounts': 3,
            'num_delinquent_accounts': 0,
            'property_value': 2000000,
            'requested_loan_amount': loan_amount,
            'requested_loan_tenure': tenure,
            'pre_approved_limit': loan_amount * 1.2,
            'monthly_income_after_emi': monthly_income - estimated_emi,
            'debt_to_income_ratio': 0.3,
            'loan_to_income_ratio': loan_amount / annual_income,
            'estimated_monthly_emi': estimated_emi,
            'emi_to_income_ratio': estimated_emi / monthly_income,
            'total_monthly_obligation': estimated_emi,
            'obligation_to_income_ratio': estimated_emi / monthly_income,
            'loan_to_asset_ratio': 0.4,
            'credit_age_months': 60,
            'income_to_loan_ratio': annual_income / loan_amount,
            'emi_affordability': monthly_income / estimated_emi if estimated_emi > 0 else 2.0,
            'asset_coverage': 1.5,
            'stability_score': 70
        }
        
        # --- Process employment type ---
        emp_type_lower = employment_type_raw.lower()
        if 'salaried' in emp_type_lower:
            employment_val = 'Salaried'
        elif 'self' in emp_type_lower or 'business' in emp_type_lower:
            employment_val = 'Self-Employed'
        else:
            employment_val = 'Salaried'  # Default
        
        # --- Categorical features ---
        categorical_defaults = {
            'gender': 'M',
            'city': 'Bangalore',
            'employment_type': employment_val,
            'education_level': 'Graduate',
            'marital_status': 'Single',
            'home_ownership': 'Rented',
            'property_type': 'Apartment'
        }
        
        # Merge all features
        model_input = {**numerical_defaults, **categorical_defaults}
        
        # Convert to DataFrame in correct order
        df_input = pd.DataFrame([model_input], columns=self.model_features_order)
        
        print(f"\nFinal preprocessed data - Sample values:")
        sample_cols = ['age', 'annual_income', 'requested_loan_amount', 'cibil_score', 'employment_type']
        for col in sample_cols:
            if col in df_input.columns:
                print(f"  {col}: {df_input[col].iloc[0]}")
        print(f"===========================\n")
        
        return df_input
    
    def perform_underwriting(self, entities: dict) -> dict:
        """
        Executes the AI + Rule-Based underwriting process.
        """
        print("\n" + "="*60)
        print("SALES ENDPOINT - Stage: ConversationStage.REJECTION_COUNSELING")
        print("User input:")
        print(f"  Approval Status: {entities.get('approval_status', 'N/A')}")
        print(f"  Risk Score: {entities.get('risk_score', 'N/A')}")
        print("="*60)
        
        # Extract values
        age = int(entities.get('age', 0))
        income = float(entities.get('annual_income') or entities.get('income') or 0)
        loan_amount = float(entities.get('requested_loan_amount') or entities.get('loan_amount') or 0)
        
        # --- PHASE 1: RULE-BASED CHECK (Hard Stops) ---
        if age < self.MIN_AGE or age > self.MAX_AGE:
            result = self._hard_reject(reason=f"Age outside policy: {self.MIN_AGE}-{self.MAX_AGE}")
            print(f"\nResponse from /sales endpoint:")
            print(f"  Stage: rejection_counseling")
            print(f"  Message preview: {result['reason']}")
            return result
        
        if income < self.MIN_INCOME:
            result = self._hard_reject(reason=f"Income below minimum policy of ₹{self.MIN_INCOME:,}")
            print(f"\nResponse from /sales endpoint:")
            print(f"  Stage: rejection_counseling")
            print(f"  Message preview: {result['reason']}")
            return result
        
        if loan_amount > self.MAX_LOAN or loan_amount < 50000:
            result = self._hard_reject(reason=f"Loan amount outside policy range (₹50,000 - ₹{self.MAX_LOAN:,})")
            print(f"\nResponse from /sales endpoint:")
            print(f"  Stage: rejection_counseling")
            print(f"  Message preview: {result['reason']}")
            return result
        
        # --- PHASE 2: AI MODEL SCORING ---
        df_input = self._preprocess_input(entities)
        
        try:
            # Get full prediction output
            prediction = self.model.predict_proba(df_input)
            print(f"\nAI Model Raw Output: {prediction}")
            
            # Extract risk score (probability of default - class 1)
            risk_score = prediction[:, 1][0]
            print(f"Extracted Risk Score: {risk_score:.3f}")
            
        except Exception as e:
            print(f"❌ AI Model Prediction Failed: {e}")
            risk_score = 0.5  # Fallback
        
        # --- PHASE 3: AI SCORE THRESHOLD RULE ---
        if risk_score > self.RISK_THRESHOLD_REJECT:
            result = self._hard_reject(
                reason=f"AI Risk Score ({risk_score:.2f}) exceeds policy threshold ({self.RISK_THRESHOLD_REJECT})"
            )
            print(f"\nResponse from /sales endpoint:")
            print(f"  Stage: rejection_counseling")
            print(f"  Message preview: {result['reason']}")
            return result
        
        # --- PHASE 4: FINAL APPROVAL ---
        interest_rate = self._mock_interest_rate(risk_score)
        
        result = {
            "approval_status": True,
            "risk_score": round(risk_score, 3),
            "interest_rate": interest_rate,
            "reason": "Approved based on policy and low AI risk score."
        }
        
        print(f"\n✓ UNDERWRITING RESULT: APPROVED")
        print(f"  Risk Score: {risk_score:.3f}")
        print(f"  Interest Rate: {interest_rate}%")
        print(f"  Reason: {result['reason']}")
        print("="*60 + "\n")
        
        return result


# # --- TESTING ---
# if __name__ == "__main__":
#     print("\n" + "="*60)
#     print("TESTING UNDERWRITING AGENT")
#     print("="*60 + "\n")
    
#     agent = UnderwritingAgent()
    
#     # Test case from your screenshot
#     test_entities = {
#         'age': 34,
#         'annual_income': 2000000,
#         'requested_loan_amount': 1000000,
#         'requested_loan_tenure': 36,
#         'cibil_score': 750,
#         'employment_type': 'Salaried'
#     }
    
#     result = agent.perform_underwriting(test_entities)
    
#     print("\n" + "="*60)
#     print("FINAL RESULT")
#     print("="*60)
#     print(f"Approval: {result['approval_status']}")
#     print(f"Risk Score: {result['risk_score']}")
#     print(f"Interest Rate: {result['interest_rate']}%")
#     print(f"Reason: {result['reason']}")
#     print("="*60) 
