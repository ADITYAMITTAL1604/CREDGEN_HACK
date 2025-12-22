import re
import string

def clean_text(text):
    """Clean and normalize text"""
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def extract_amount(text):
    """Extract loan amount from text
    Examples: '5 lakhs', '₹500000', '5L', 'five lakh'
    """
    text = text.lower()
    
    # Pattern 1: Direct numbers with ₹ or Rs
    pattern1 = r'[₹rs.\s]*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lakh|lac|l)?'
    
    # Pattern 2: Word form
    word_to_num = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    # Try numeric pattern
    matches = re.findall(pattern1, text)
    for match in matches:
        amount = float(match.replace(',', ''))
        # Check if lakhs mentioned
        if 'lakh' in text or 'lac' in text or 'l' in text:
            amount *= 100000
        return int(amount)
    
    # Try word form
    for word, num in word_to_num.items():
        if word in text and ('lakh' in text or 'lac' in text):
            return num * 100000
    
    return None

def extract_tenure(text):
    """Extract loan tenure in months
    Examples: '3 years', '36 months', '5Y'
    """
    text = text.lower()
    
    # Pattern for years
    year_pattern = r'(\d+)\s*(?:year|yr|y)'
    year_match = re.search(year_pattern, text)
    if year_match:
        return int(year_match.group(1)) * 12
    
    # Pattern for months
    month_pattern = r'(\d+)\s*(?:month|mon|m)'
    month_match = re.search(month_pattern, text)
    if month_match:
        return int(month_match.group(1))
    
    return None

def extract_age(text):
    """Extract age from text"""
    pattern = r'\b(\d{2})\b(?:\s*(?:year|yr|y|age))?'
    matches = re.findall(pattern, text)
    for match in matches:
        age = int(match)
        if 18 <= age <= 80:  # Reasonable age range
            return age
    return None

def extract_income(text):
    """Extract income from text
    Examples: '8 LPA', '80k per month', '₹960000 yearly'
    """
    text = text.lower()
    
    # Pattern for LPA (Lakhs Per Annum)
    lpa_pattern = r'(\d+(?:\.\d+)?)\s*(?:lpa|lakh per annum)'
    lpa_match = re.search(lpa_pattern, text)
    if lpa_match:
        return int(float(lpa_match.group(1)) * 100000)
    
    # Pattern for monthly with k
    monthly_k_pattern = r'(\d+)k?\s*(?:per month|pm|monthly|/month)'
    monthly_match = re.search(monthly_k_pattern, text)
    if monthly_match:
        monthly = int(monthly_match.group(1)) * 1000
        return monthly * 12
    
    # Direct annual amount
    annual_pattern = r'[₹rs.\s]*(\d+(?:,\d+)*)\s*(?:yearly|annual|per annum|pa)'
    annual_match = re.search(annual_pattern, text)
    if annual_match:
        return int(annual_match.group(1).replace(',', ''))
    
    return None

def extract_name(text):
    """Extract name using simple patterns"""
    # Pattern: "I'm NAME" or "My name is NAME"
    pattern1 = r"(?:i'm|i am|my name is|this is)\s+([a-z]+(?:\s+[a-z]+)?)"
    match = re.search(pattern1, text.lower())
    if match:
        name = match.group(1)
        return name.title()
    
    return None

def extract_pan(text):
    """Extract PAN card number"""
    pattern = r'\b[A-Z]{5}\d{4}[A-Z]\b'
    match = re.search(pattern, text.upper())
    return match.group(0) if match else None

def extract_aadhaar(text):
    """Extract Aadhaar number"""
    pattern = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
    match = re.search(pattern, text)
    if match:
        return match.group(0).replace('-', '').replace(' ', '')
    return None

def extract_pincode(text):
    """Extract 6-digit pincode"""
    pattern = r'\b\d{6}\b'
    match = re.search(pattern, text)
    return match.group(0) if match else None

def extract_employment_type(text):
    """Extract employment type"""
    text = text.lower()
    if any(word in text for word in ['salaried', 'salary', 'employee', 'job']):
        return 'salaried'
    elif any(word in text for word in ['self employed', 'self-employed', 'business', 'entrepreneur']):
        return 'self_employed'
    elif 'professional' in text:
        return 'professional'
    return None

def extract_purpose(text):
    """Extract loan purpose"""
    text = text.lower()
    purposes = {
        'home': ['home', 'house', 'property', 'renovation', 'repair'],
        'education': ['education', 'study', 'college', 'university', 'course'],
        'business': ['business', 'startup', 'venture', 'company'],
        'medical': ['medical', 'health', 'hospital', 'treatment'],
        'personal': ['personal', 'family', 'wedding', 'travel']
    }
    
    for purpose, keywords in purposes.items():
        if any(keyword in text for keyword in keywords):
            return purpose
    return 'personal'  # default

def validate_amount(amount):
    """Validate loan amount against constraints"""
    from utils.config import MIN_LOAN_AMOUNT, MAX_LOAN_AMOUNT
    if amount and MIN_LOAN_AMOUNT <= amount <= MAX_LOAN_AMOUNT:
        return True
    return False

def validate_age(age):
    """Validate age against constraints"""
    from utils.config import MIN_AGE, MAX_AGE
    if age and MIN_AGE <= age <= MAX_AGE:
        return True
    return False

def validate_tenure(tenure):
    """Validate tenure against constraints"""
    from utils.config import MIN_TENURE, MAX_TENURE
    if tenure and MIN_TENURE <= tenure <= MAX_TENURE:
        return True
    return False
