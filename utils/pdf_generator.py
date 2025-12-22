import asyncio
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from datetime import date
from reportlab.lib.units import inch
from pathlib import Path
import os

# Removed dependency on pdfrw (PdfReader, PdfWriter, PageMerge)
# Removed dependency on custom fonts (Arial.ttf) and PdfMetrics
# This code generates the entire PDF content from scratch using reportlab's built-in fonts.

# --- 1. Data Mapping and Pre-Calculation Utility ---


def get_pdf_input_details(state: dict) -> dict:
    """
    Maps the Master Agent's state data to the keys required by the PDF generator.
    """
    entities = state.get('entities', {})
    loan_amount = entities.get('loan_amount', 0)
    
    # Policy Rule: Calculate 1% processing fee of the loan amount
    processing_charges = loan_amount * 0.01

    details = {
        # Required KYC/Application fields
        'cust_name': entities.get('name', 'Loan Applicant'),
        'cust_add': f"{entities.get('address', 'Address N/A')}, Pincode: {entities.get('pincode', 'N/A')}",
        'amt': loan_amount,
        'tenure': entities.get('tenure', 36), # in months
        
        # Offer details
        'roi': state.get('interest_rate', 15.0),
        'processing_charges': processing_charges,
        
        # Mocked fields (if not collected by Master Agent)
        'coborrower': entities.get('coborrower', 'NIL'), 
    }
    return details

# --- 2. Core Asynchronous PDF Generation (Full Generation from Scratch) ---

async def _async_gen_sl(cust_details: dict) -> str:
    """
    Generates the entire PDF Sanction Letter from scratch using reportlab.
    This replaces the original file-dependent 'gen_sl' function.
    """
    # Define output file path relative to the project structure
    current_dir = Path(__file__).resolve()
    # Adjust root path assumption to match the project structure (backend/pdf_generator.py -> project_root/data)
    root_path = current_dir.parent.parent 
    safe_name = cust_details['cust_name'].replace(' ', '_').replace('.', '')
    
    # Ensure the 'data' directory exists for the output file
    data_dir = root_path / "data"
    data_dir.mkdir(exist_ok=True)
    out_path = data_dir / f"Sanction_Letter_{safe_name}_{date.today()}.pdf"
    
    date_issue = date.today().strftime("%B %d, %Y")
    
    def create_sanction_letter():
        c = canvas.Canvas(str(out_path), pagesize=A4)
        width, height = A4
        
        # --- HEADER & TITLE ---
        c.setFont("Times-Bold", 16)
        c.drawString(inch, height - inch, "CredGen Financial Services")
        c.setFont("Times-Bold", 14)
        c.drawString(inch, height - inch - 0.3*inch, "Personal Loan Sanction Letter")

        # --- DATE & ADDRESS ---
        c.setFont("Times-Roman", 11)
        c.drawString(inch, height - 2*inch, f"Date: {date_issue}")
        c.drawString(inch, height - 2.5*inch, f"To,")
        c.drawString(inch, height - 2.7*inch, f"Customer: {cust_details['cust_name']}")
        c.drawString(inch, height - 2.9*inch, f"Address: {cust_details['cust_add']}")

        # --- SALUTATION ---
        c.drawString(inch, height - 3.5*inch, f"Dear {cust_details['cust_name']},")
        c.drawString(inch, height - 3.8*inch, "We are pleased to inform you that your application for a Personal Loan has been approved.")
        
        # --- LOAN DETAILS TABLE/SECTION ---
        y_start = height - 4.5*inch
        col1_x = inch * 1.5
        col2_x = inch * 5.0
        row_height = 0.3*inch

        c.setFont("Times-Bold", 11)
        c.drawString(col1_x, y_start, "Loan Parameter")
        c.drawString(col2_x, y_start, "Sanctioned Value")
        
        c.line(inch, y_start - 0.1*inch, width - inch, y_start - 0.1*inch)
        
        y = y_start - row_height
        c.setFont("Times-Roman", 11)
        
        # Loan Amount
        c.drawString(col1_x, y, "Loan Amount")
        c.drawString(col2_x, y, f"Rs. {cust_details['amt']:,.0f} ONLY")
        y -= row_height

        # Tenure
        c.drawString(col1_x, y, "Tenure")
        c.drawString(col2_x, y, f"{cust_details['tenure']} Months")
        y -= row_height

        # Interest Rate (ROI)
        c.drawString(col1_x, y, "Interest Rate (ROI)")
        c.drawString(col2_x, y, f"{cust_details['roi']:.2f} % per annum")
        y -= row_height

        # Processing Charges
        c.drawString(col1_x, y, "Processing Charges (1%)")
        c.drawString(col2_x, y, f"Rs. {cust_details['processing_charges']:,.2f}")
        y -= row_height
        
        # Co-Borrower
        c.drawString(col1_x, y, "Co-Borrower")
        c.drawString(col2_x, y, f"{cust_details['coborrower']}")
        y -= row_height
        
        c.line(inch, y + 0.1*inch, width - inch, y + 0.1*inch)

        # --- CLOSING ---
        c.setFont("Times-Roman", 10)
        c.drawString(inch, y - row_height, "Please sign and return a copy of this letter within 7 days to accept the terms.")
        c.drawString(inch, y - 1.5*inch, "Thank you for choosing CredGen.")
        c.drawString(inch, y - 2.5*inch, "Sincerely,")
        
        c.setFont("Times-Roman", 11)
        c.drawString(inch, y - 3*inch, "CredGen Agent Team")

        c.save()
        
    await asyncio.to_thread(create_sanction_letter)

    return str(out_path)

# --- 3. Synchronous Wrapper for Flask/app.py ---

def generate_sanction_letter(master_agent_state: dict) -> str:
    """
    Synchronous wrapper to be called by backend/app.py.
    It prepares the data and executes the async PDF generation.
    """
    cust_details = get_pdf_input_details(master_agent_state)
    
    
    try:
        # Get the current event loop. If one is running, use it. Otherwise, create and run a new one.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        pdf_path = loop.run_until_complete(_async_gen_sl(cust_details))
            
        return pdf_path
        
    except Exception as e:
        print(f"PDF Generation Error: {e}")
        return f"ERROR: Failed to generate PDF: {e}"
