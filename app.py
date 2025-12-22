from flask import Flask, request, jsonify, send_from_directory, render_template, session, redirect, url_for
from flask_cors import CORS
from dotenv import load_dotenv
import os
import uuid
import time
from datetime import datetime
import threading
import json
from werkzeug.utils import secure_filename

# Import the core agents
from agents.master_agent import MasterAgent, ConversationStage, IntentType
from agents.underwriting_agent import UnderwritingAgent
from agents.sales_agent import SalesAgent
from agents.fraud_agent import FraudAgent
from utils.pdf_generator import generate_sanction_letter as generate_sanction_pdf
from models.gemini_service import GeminiService
from models.openrouter_service import OpenRouterService

# Load environment variables
load_dotenv()

# --- 1. Initialization ---
app = Flask(__name__)
CORS(app)

# Basic config and paths (for admin system)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csv")
os.makedirs(CSV_DIR, exist_ok=True)

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

APPLICATIONS_CSV = os.path.join(CSV_DIR, "applications.csv")
CHAT_LOGS_CSV = os.path.join(CSV_DIR, "chat_logs.csv")
TUNING_CSV = os.path.join(CSV_DIR, "tuning_content.csv")
ADMINS_CSV = os.path.join(CSV_DIR, "admin_users.csv")

app.secret_key = os.environ.get("APP_SECRET_KEY", "credgen")
app.config["UPLOAD_FOLDER"] = UPLOADS_DIR
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# Session management with expiration
user_sessions = {}
SESSION_TIMEOUT = 1800  # 30 minutes in seconds

# Initialize agents
underwriting_agent = UnderwritingAgent()
sales_agent = SalesAgent()
fraud_agent = FraudAgent()

# Initialize Active LLM Service
llm_provider = os.getenv("LLM_PROVIDER", "openrouter").lower().strip()
llm_service = None

if llm_provider == "openrouter":
    llm_service = OpenRouterService()
    print(f"Initialized OpenRouter Service (Provider: {llm_provider})")
else:
    llm_service = GeminiService()
    print(f"Initialized Gemini Service (Provider: {llm_provider})")

# Cleanup thread for expired sessions
def cleanup_sessions():
    """Periodically clean up expired sessions"""
    while True:
        time.sleep(300)  # Run every 5 minutes
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session_data in user_sessions.items():
            if current_time - session_data.get('last_activity', 0) > SESSION_TIMEOUT:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del user_sessions[session_id]
            print(f"Cleaned up expired session: {session_id}")

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
cleanup_thread.start()

# --- Utility Functions ---

def get_session_id(request):
    """Retrieve or create a session ID with validation."""
    session_id = request.headers.get('X-Session-ID')
    
    if not session_id:
        # Generate new session ID
        session_id = f"session_{uuid.uuid4().hex[:16]}"
    
    return session_id

def update_session_activity(session_id):
    """Update last activity time for session."""
    if session_id in user_sessions:
        user_sessions[session_id]['last_activity'] = time.time()

def initialize_user_session(session_id):
    """Initialize a new session for a user."""
    if session_id not in user_sessions:
        user_sessions[session_id] = {
            'master_agent': MasterAgent(),  # Create new instance per session
            'last_activity': time.time(),
            'created_at': datetime.now().isoformat(),
            'interaction_count': 0,
            'workflow_history': []
        }
    
    update_session_activity(session_id)
    return user_sessions[session_id]

def generate_sanction_letter_text(state: dict) -> dict:
    """Generate the final sanction letter with all details."""
    entities = state.get('entities', {})
    current_offer = state.get('current_offer', {})
    
    # Extract data with fallbacks
    loan_amount = entities.get('loan_amount', 0)
    interest_rate = current_offer.get('interest_rate', state.get('interest_rate', 'N/A'))
    
    if isinstance(interest_rate, (int, float)):
        rate_str = f"{interest_rate:.2f}%"
        emi = current_offer.get('monthly_emi', calculate_emi(loan_amount, interest_rate, entities.get('tenure', 60)))
    else:
        rate_str = str(interest_rate)
        emi = "N/A"
    
    name = entities.get('name', 'Applicant')
    tenure = entities.get('tenure', 60)
    date_today = datetime.now().strftime("%B %d, %Y")

    # Format EMI safely for display
    if isinstance(emi, (int, float)):
        emi_str = f"{emi:,.2f}"
    else:
        emi_str = str(emi)
    
    letter_content = f"""
CREDGEN FINANCIAL SERVICES
==========================
SANCTION LETTER

Date: {date_today}
Sanction Letter No: SL-{uuid.uuid4().hex[:8].upper()}

Dear {name},

We are pleased to inform you that your loan application has been APPROVED.

Loan Details:
-------------
• Sanctioned Amount: ₹{loan_amount:,}
• Interest Rate: {rate_str}
• Loan Tenure: {tenure} months
• Monthly EMI: ₹{emi_str}
• Processing Fee: ₹{max(1000, loan_amount * 0.01):,}
• Disbursement Date: Within 3 working days

Terms & Conditions:
-------------------
1. This sanction is valid for 30 days from the date of issue.
2. Final disbursement is subject to document verification.
3. Rate of interest is subject to change as per market conditions.
4. Penalty for late payment: 2% per month.

Please sign and return this letter to proceed with disbursement.

For CredGen Financial Services,
[Authorized Signatory]

This is a computer-generated letter and does not require a signature.
"""
    
    return {
        'content': letter_content,
        'metadata': {
            'sanction_id': f"SL-{uuid.uuid4().hex[:8].upper()}",
            'date': date_today,
            'applicant': name,
            'amount': loan_amount,
            'interest_rate': rate_str,
            'tenure': tenure
        }
    }

def calculate_emi(principal, rate, tenure_months):
    """Calculate EMI (simplified)."""
    if tenure_months <= 0:
        return 0
    monthly_rate = rate / 12 / 100
    emi = principal * monthly_rate * (1 + monthly_rate) ** tenure_months / ((1 + monthly_rate) ** tenure_months - 1)
    return round(emi, 2)

def get_bank_context():
    """Retrieve and concatenate all tuning context from CSV."""
    try:
        rows = read_csv(TUNING_CSV)
        # Extract content field, filter empty
        contents = [r.get('content', '').strip() for r in rows if r.get('content', '').strip()]
        if contents:
            return "\n\n".join(contents)
    except Exception as e:
        print(f"Error loading bank context: {e}")
    return ""

def determine_worker_from_stage(current_stage, intent=None):
    """Determine which worker to call based on current stage."""
    worker_map = {
        ConversationStage.FRAUD_CHECK: "fraud",
        ConversationStage.UNDERWRITING: "underwriting",
        ConversationStage.OFFER_PRESENTATION: "sales",
        ConversationStage.DOCUMENTATION: "documentation",
        ConversationStage.REJECTION_COUNSELING: "sales"
    }
    
    return worker_map.get(current_stage, "none")

def get_workflow_stage_details(stage):
    """Get details about a workflow stage."""
    stage_details = {
        ConversationStage.COLLECTING_DETAILS: {
            "name": "Basic Details Collection",
            "description": "Collecting loan requirements and personal information",
            "progress": 20,
            "next": "KYC Collection"
        },
        ConversationStage.KYC_COLLECTION: {
            "name": "KYC Verification",
            "description": "Collecting identification documents for verification",
            "progress": 40,
            "next": "Fraud Detection"
        },
        ConversationStage.FRAUD_CHECK: {
            "name": "Fraud Detection",
            "description": "Running security checks and verification",
            "progress": 60,
            "next": "Underwriting"
        },
        ConversationStage.UNDERWRITING: {
            "name": "Underwriting",
            "description": "Assessing credit risk and loan eligibility",
            "progress": 80,
            "next": "Offer Presentation"
        },
        ConversationStage.OFFER_PRESENTATION: {
            "name": "Offer Presentation",
            "description": "Presenting loan terms and conditions",
            "progress": 90,
            "next": "Documentation"
        },
        ConversationStage.DOCUMENTATION: {
            "name": "Documentation",
            "description": "Generating sanction letter and final documents",
            "progress": 100,
            "next": "Completion"
        }
    }
    
    return stage_details.get(stage, {"name": "Unknown", "progress": 0})

# --- Admin/CSV Utilities ---
def ensure_csv(file_path, headers):
    exists = os.path.exists(file_path)
    if not exists:
        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            import csv as _csv
            writer = _csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

def append_csv(file_path, row, headers):
    import csv as _csv
    ensure_csv(file_path, headers)
    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=headers)
        writer.writerow(row)

def read_csv(file_path):
    import csv as _csv
    if not os.path.exists(file_path):
        return []
    with open(file_path, mode="r", newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        return list(reader)

def log_chat_event(session_id, event_type, payload):
    import csv as _csv
    headers = [
        "timestamp",
        "session_id",
        "event_type",
        "message_role",
        "message_text",
        "status",
        "details",
    ]
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "event_type": event_type,
        "message_role": payload.get("role"),
        "message_text": payload.get("text"),
        "status": payload.get("status"),
        "details": json.dumps(payload.get("details", {}), ensure_ascii=False),
    }
    append_csv(CHAT_LOGS_CSV, row, headers)

ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def seed_default_admin():
    headers = ["username", "password"]
    ensure_csv(ADMINS_CSV, headers)
    admins = read_csv(ADMINS_CSV)
    if not admins:
        append_csv(ADMINS_CSV, {"username": "admin", "password": "admin123"}, headers)

seed_default_admin()

def init_csv_files():
    """Ensure core CSVs exist with headers so they are visible upfront."""
    ensure_csv(APPLICATIONS_CSV, [
        'timestamp', 'session_id', 'full_name', 'phone', 'email',
        'city', 'loan_amount', 'status', 'rejection_reason', 'attached_files'
    ])
    ensure_csv(CHAT_LOGS_CSV, [
        'timestamp', 'session_id', 'event_type', 'message_role',
        'message_text', 'status', 'details'
    ])
    ensure_csv(TUNING_CSV, ['timestamp', 'admin', 'type', 'content', 'filename'])

# Create empty CSV files with headers at startup
init_csv_files()

# --- Frontend Routes ---

@app.route('/')
def index():
    """Serve the main frontend page."""
    return send_from_directory('frontend', 'index.html')

@app.route('/widget.html')
def widget():
    """Serve the widget page for 3rd party embedding."""
    return send_from_directory('frontend', 'widget.html')

@app.route('/frontend/<path:filename>')
def frontend_files(filename):
    """Serve static frontend files (CSS, JS, etc.)."""
    return send_from_directory('frontend', filename)

# --- API Endpoints ---

@app.route('/chat', methods=['POST'])
def chat():
    """
    Primary conversational endpoint with OpenRouter enabled mode.
    """
    try:
        session_id = get_session_id(request)

        # Support both JSON and multipart/form-data
        user_input = ''
        if request.content_type and 'multipart/form-data' in request.content_type:
            user_input = (request.form.get('message') or '').strip()
        else:
            data = request.get_json(silent=True) or {}
            user_input = (data.get('message') or '').strip()

        if not user_input:
            return jsonify({
                'message': 'Please provide a message.',
                'error': 'empty_input'
            }), 400
        
        # Initialize or retrieve session
        log_chat_event(session_id, 'message', {
            'role': 'user',
            'text': user_input,
            'status': None,
            'details': {}
        })

        session = initialize_user_session(session_id)
        session['interaction_count'] += 1
        
        # Get the user's master agent instance
        user_master_agent = session['master_agent']
        
        # --- LLM MODE HANDLING ---
        gemini_mode = os.getenv("LLM_MODE", "enabled").lower().strip()
        
        # Load base prompt and dynamic context
        base_prompt = os.getenv("LLM_SYSTEM_PROMPT", "You are CredGen AI, an intelligent loan assistant.")
        bank_context = get_bank_context()
        
        if bank_context:
            system_prompt = f"{base_prompt}\n\n[BANK SPECIFIC CONTEXT]\n{bank_context}\n[END CONTEXT]"
        else:
            system_prompt = base_prompt

        response = None
        
        # Mode 1: Enabled (OpenRouter Only / Primary)
        if gemini_mode == "enabled":
            try:
                # Prepare context with current workflow state
                current_stage = user_master_agent.state["stage"]
                workflow_progress = user_master_agent.state.get("workflow_progress", 0)
                collected_entities = {k: v for k, v in user_master_agent.state["entities"].items() if v}
                
                # Build enhanced system prompt with workflow context
                stage_details = get_workflow_stage_details(current_stage)
                
                workflow_prompt = f"""
                CURRENT WORKFLOW STATE:
                - Stage: {stage_details.get('name', current_stage.value)}
                - Progress: {workflow_progress}%
                - Next Step: {stage_details.get('next', 'Completion')}
                - Collected Information: {json.dumps(collected_entities, default=str)}
                - Missing Basic Details: {list(user_master_agent.state['missing_fields'])}
                - Missing KYC Details: {list(user_master_agent.state['missing_kyc_fields'])}
                
                WORKFLOW SEQUENCE (MUST FOLLOW THIS ORDER):
                1. BASIC DETAILS: loan_amount, tenure, age, income, name, employment_type, purpose
                2. KYC COLLECTION: pan, aadhaar, address, pincode
                3. FRAUD DETECTION: (automated after KYC)
                4. UNDERWRITING: (automated after fraud check)
                5. OFFER PRESENTATION: (after approval)
                6. DOCUMENTATION: (after acceptance)
                
                Your response should:
                1. Be natural and conversational
                2. Guide the user through the current stage
                3. Ask for missing information
                4. Confirm when a stage is complete
                """
                
                full_system_prompt = f"{system_prompt}\n\n{workflow_prompt}"
                
                # Get response from OpenRouter
                llm_resp = llm_service.generate_response(user_input, full_system_prompt)
                
                if llm_resp.get("status") == "success":
                    # Extract entities if provided by LLM
                    if "extracted_entities" in llm_resp:
                        entities = llm_resp["extracted_entities"]
                        # Clean None values
                        entities = {k: v for k, v in entities.items() if v is not None}
                        if entities:
                            # Update master agent state with extracted entities
                            intent = IntentType.PROVIDE_INFO if entities else IntentType.UNCLEAR
                            user_master_agent.update_state(entities, intent)
                    
                    # Determine next worker based on current stage
                    worker = determine_worker_from_stage(current_stage)
                    
                    # Map worker to action
                    action_map = {
                        "fraud": "call_fraud_api",
                        "underwriting": "call_underwriting_api",
                        "sales": "call_sales_api",
                        "documentation": "call_documentation_api"
                    }
                    
                    response = {
                        "message": llm_resp.get("message", ""),
                        "suggestions": llm_resp.get("suggestions", []),
                        "worker": worker,
                        "action": action_map.get(worker, "none"),
                        "intent": "llm_response",
                        "stage": user_master_agent.state["stage"].value,
                        "stage_name": stage_details.get("name", ""),
                        "workflow_progress": user_master_agent.state.get("workflow_progress", 0),
                        "session_id": session_id,
                        "entities_collected": collected_entities,
                        "missing_fields": list(user_master_agent.state['missing_fields']),
                        "missing_kyc_fields": list(user_master_agent.state['missing_kyc_fields']),
                        "terminate": False
                    }
                    
                    # Special handling for offer acceptance
                    if "accept" in user_input.lower() and current_stage == ConversationStage.OFFER_PRESENTATION:
                        user_master_agent.set_offer_accepted(True)
                        worker = "sales"
                        action = "call_sales_api"
                        response['worker'] = worker
                        response['action'] = action
                        response['message'] = "Processing your acceptance..."
                        
                else:
                    # Fallback to backend on AI failure
                    print(f"OpenRouter failed, falling back: {llm_resp.get('error', 'Unknown error')}")
                    response = user_master_agent.handle(user_input)
                    
            except Exception as e:
                print(f"OpenRouter Enabled Mode Error: {e}")
                response = user_master_agent.handle(user_input)

        # Mode 2: Hybrid (Orchestration)
        elif gemini_mode == "hybrid":
            try:
                # 1. Determine intent using backend
                intent, confidence = user_master_agent.detect_intent(user_input)
                
                # 2. Define "generative" vs "process" intents
                generative_intents = [
                    IntentType.GREETING, 
                    IntentType.HELP_GENERAL, 
                    IntentType.UNCLEAR,
                    IntentType.EXIT
                ]
                
                if intent in generative_intents:
                    # Use AI for natural language generation
                    context_prompt = system_prompt + f"\n[Context: Detected Intent '{intent.value}', Stage '{user_master_agent.state['stage'].value}']"
                    
                    llm_resp = llm_service.generate_response(user_input, context_prompt)
                    
                    if llm_resp.get("status") == "success":
                        response = {
                            "message": llm_resp.get("message", ""),
                            "suggestions": llm_resp.get("suggestions", []),
                            "worker": "none",
                            "action": "none",
                            "intent": intent.value,
                            "stage": user_master_agent.state["stage"].value,
                            "session_id": session_id
                        }
                    else:
                        response = user_master_agent.handle(user_input)
                else:
                    # Deterministic tasks -> Backend
                    response = user_master_agent.handle(user_input)

            except Exception as e:
                print(f"OpenRouter Hybrid Mode Error: {e}")
                response = user_master_agent.handle(user_input)

        # Mode 3: Disabled (Default to existing backend)
        else:
            response = user_master_agent.handle(user_input)
            
        # Ensure response is set (safety fallback)
        if response is None:
            response = user_master_agent.handle(user_input)
        
        # Update session state
        session['last_state'] = user_master_agent.state.copy()
        
        # Add workflow history
        if 'workflow_history' not in session:
            session['workflow_history'] = []
        
        session['workflow_history'].append({
            'timestamp': datetime.now().isoformat(),
            'stage': user_master_agent.state["stage"].value,
            'progress': user_master_agent.state.get("workflow_progress", 0),
            'user_input': user_input[:100],
            'response': response.get('message', '')[:100]
        })
        
        # Add session info to response
        response['session_id'] = session_id
        response['interaction_count'] = session['interaction_count']

        # Log assistant response
        try:
            log_chat_event(session_id, 'message', {
                'role': 'assistant',
                'text': response.get('message'),
                'status': None,
                'details': {
                    'worker': response.get('worker'),
                    'action': response.get('action'),
                    'stage': response.get('stage'),
                    'progress': response.get('workflow_progress')
                }
            })
        except Exception:
            pass

        resp = jsonify(response)
        resp.headers['X-Session-ID'] = session_id
        return resp
        
    except Exception as e:
        print(f"Error in /chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'message': 'Sorry, I encountered an error. Please try again.',
            'error': 'server_error',
            'worker': 'none'
        }), 500

@app.route('/underwrite', methods=['POST'])
def underwrite():
    """
    Worker endpoint for underwriting process.
    """
    try:
        session_id = request.headers.get('X-Session-ID')

        if not session_id or session_id not in user_sessions:
            return jsonify({'error': 'Invalid or expired session.'}), 400
        
        session = user_sessions[session_id]
        update_session_activity(session_id)
        
        user_master_agent = session['master_agent']
        current_state = user_master_agent.state
        
        # Step 1: Check if KYC is complete (required before underwriting)
        if current_state['missing_kyc_fields']:
            return jsonify({
                'message': 'Please complete KYC details first.',
                'error': 'kyc_incomplete',
                'worker': 'none',
                'next_action': 'collect_kyc'
            }), 400
        
        # Step 2: Check fraud status (must pass before underwriting)
        if not current_state.get('fraud_check_passed'):
            return jsonify({
                'message': 'Fraud check must be completed first.',
                'error': 'fraud_check_pending',
                'worker': 'fraud',
                'action': 'call_fraud_api'
            }), 400
        
        # Step 3: Proceed with underwriting
        underwriting_result = underwriting_agent.perform_underwriting(
            current_state['entities']
        )
        
        # Step 4: Update master agent with underwriting result
        user_master_agent.set_underwriting_result(
            risk_score=underwriting_result['risk_score'],
            approval_status=underwriting_result['approval_status'],
            interest_rate=underwriting_result.get('interest_rate', 12.5)
        )
        
        # Step 5: Generate response based on result
        if underwriting_result['approval_status']:
            response = {
                'message': '✅ Your application has been approved! Generating loan offer...',
                'approval_status': True,
                'risk_score': underwriting_result['risk_score'],
                'interest_rate': underwriting_result.get('interest_rate', 12.5),
                'worker': 'sales',
                'action': 'call_sales_api',
                'stage': 'offer_presentation',
                'workflow_progress': 80
            }
        else:
            response = {
                'message': '❌ Unfortunately, your application was not approved at this time.',
                'approval_status': False,
                'reason': underwriting_result.get('reason', 'risk_assessment'),
                'worker': 'sales',  # Route to sales for counseling
                'action': 'call_sales_api',
                'stage': 'rejection_counseling'
            }
        
        # Update session
        session['underwriting_result'] = underwriting_result
        session['last_state'] = user_master_agent.state.copy()

        # Log status change
        try:
            log_chat_event(session_id, 'status_change', {
                'role': 'system',
                'text': None,
                'status': 'approved' if underwriting_result['approval_status'] else 'rejected',
                'details': {
                    'risk_score': underwriting_result.get('risk_score'),
                    'interest_rate': underwriting_result.get('interest_rate')
                }
            })
        except Exception:
            pass
        
        response['session_id'] = session_id
        resp = jsonify(response)
        resp.headers['X-Session-ID'] = session_id
        return resp
        
    except Exception as e:
        print(f"Error in /underwrite: {e}")
        return jsonify({
            'error': 'underwriting_failed',
            'message': 'Underwriting process failed. Please try again.'
        }), 500

@app.route('/sales', methods=['POST'])
def sales_negotiate():
    """
    Worker endpoint for sales and negotiation.
    Handles both OFFER_PRESENTATION and REJECTION_COUNSELING.
    """
    try:
        session_id = request.headers.get('X-Session-ID')
        
        if not session_id or session_id not in user_sessions:
            return jsonify({'error': 'Invalid or expired session.'}), 400
        
        session = user_sessions[session_id]
        update_session_activity(session_id)
        
        user_master_agent = session['master_agent']
        current_state = user_master_agent.state
        current_stage = current_state.get('stage')
        
        # Get request data
        request_data = request.get_json(silent=True) or {}
        user_input = request_data.get('message', '')
        
        print(f"\n{'='*60}")
        print(f"SALES ENDPOINT - Stage: {current_stage}")
        print(f"User input: {user_input}")
        print(f"Approval Status: {current_state.get('approval_status')}")
        print(f"Risk Score: {current_state.get('risk_score')}")
        print(f"{'='*60}")
        
        response = {}
        # Handle based on current stage
        if current_stage == ConversationStage.OFFER_PRESENTATION:
            # Check if user wants to negotiate
            if not user_master_agent.state.get('offer'):
                # FIRST ENTRY: show offer
                sales_offer = sales_agent.generate_offer(master_agent_state=current_state, negotiation_request=False)
                user_master_agent.set_offer(sales_offer)

                response = {
                    **sales_offer,
                    'action': 'wait_for_offer_decision',
                    'stage': 'offer_presentation'
                }
            else:
                negotiation_requested = any(word in user_input.lower() for word in 
                                        ['negotiate', 'lower', 'reduce', 'better rate', 'discount'])
                
                accept_keywords = ['yes', 'accept', 'proceed', 'okay', 'agree', 'approved', 'i accept']
                offer_accepted = any(word in user_input.lower() for word in accept_keywords)
                print(f"Negotiation requested: {negotiation_requested}")
                print(f"Offer Accepted: {offer_accepted}")

                if offer_accepted:
                    print('User accepted the offer. Moving to documentation.')
                    user_master_agent.set_offer_accepted(True)

                    existing_offer = user_master_agent.state.get('offer')
                    user_master_agent.set_offer(existing_offer)
                    response = {
                        'message': '✅ Excellent! Your loan offer has been accepted. Generating your sanction letter now...',
                        'session_id': session_id,
                        'stage': 'documentation',
                        'workflow_progress': 95,
                        'worker': 'documentation',
                        'action': 'call_documentation_api',
                        'offer_accepted': True
                    }
                    user_master_agent.state['stage'] = ConversationStage.DOCUMENTATION
                    user_master_agent.state['offer_accepted'] = True
                elif negotiation_requested:
                    # Generate loan offer (standard or negotiated)
                    sales_offer = sales_agent.generate_offer(
                        master_agent_state=current_state,
                        negotiation_request=negotiation_requested
                    )
                    
                    # Update master agent with offer
                    user_master_agent.set_offer(sales_offer)
                    
                    response = {
                        **sales_offer,
                        'session_id': session_id,
                        'stage': 'offer_presentation',
                        'workflow_progress': 90,
                        'action': 'wait_for_offer_decision'
                    }
                else:
                    response = {
                        'message': "Please confirm acceptance or request a better offer.",
                        'stage': 'offer_presentation',
                        'action': 'wait_for_offer_decision'
                    }
            
        elif current_stage == ConversationStage.REJECTION_COUNSELING:
            print("In REJECTION_COUNSELING mode")
            
            # Check if user accepts alternative offer
            accept_keywords = ['yes', 'accept', 'proceed', 'okay', 'agree', 'alternative']
            accept_alternative = any(word in user_input.lower() for word in accept_keywords)
            
            if accept_alternative:
                print("User accepted alternative offer")
                # Generate alternative offer
                sales_offer = sales_agent.generate_offer(
                    master_agent_state=current_state,
                    negotiation_request=False
                )
                
                user_master_agent.set_offer(sales_offer)
                user_master_agent.state['stage'] = ConversationStage.OFFER_PRESENTATION
                user_master_agent.state['approval_status'] = True  # Mark as approved for alternative
                
                response = {
                    **sales_offer,
                    'session_id': session_id,
                    'stage': 'offer_presentation',
                    'workflow_progress': 90,
                    'action': 'wait_for_offer_decision',
                    'message': "Great! Here's your alternative loan offer:"
                }
            else:
                # Provide counseling
                counseling_response = sales_agent.provide_counseling(current_state)
                
                response = {
                    'message': counseling_response,
                    'session_id': session_id,
                    'stage': 'rejection_counseling',
                    'workflow_progress': 70,
                    'worker': 'sales',
                    'action': 'provide_counseling',
                    'offer_available': True,
                    'suggestions': [
                        {'text': 'Yes, show me alternative amount', 'action': 'accept_alternative'},
                        {'text': 'No, I\'ll improve my profile first', 'action': 'decline_alternative'},
                        {'text': 'Explain why I was rejected', 'action': 'explain_rejection'}
                    ]
                }
        
        else:
            # Default response if stage is not recognized
            response = {
                'message': 'I need more information to provide an offer. Let me check your application status.',
                'session_id': session_id,
                'worker': 'none',
                'action': 'check_status'
            }
        
        # Update session
        session['last_state'] = user_master_agent.state.copy()
        
        print(f"\nResponse from /sales endpoint:")
        print(f"Stage: {response.get('stage')}")
        print(f"Message preview: {response.get('message', '')[:100]}...")
        print(f"{'='*60}\n")

        resp = jsonify(response)
        resp.headers['X-Session-ID'] = session_id
        return resp
        
    except Exception as e:
        print(f"Error in /sales: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'sales_processing_failed',
            'message': 'Failed to process sales request.',
            'details': str(e)
        }), 500

@app.route('/fraud', methods=['POST'])
def fraud_check():
    """
    Dedicated endpoint for fraud detection.
    """
    try:
        session_id = request.headers.get('X-Session-ID')
        
        if not session_id or session_id not in user_sessions:
            return jsonify({'error': 'Invalid or expired session.'}), 400
        
        session = user_sessions[session_id]
        update_session_activity(session_id)
        
        user_master_agent = session['master_agent']
        current_state = user_master_agent.state
        
        # Check if basic details are complete
        if current_state['missing_fields']:
            return jsonify({
                'message': 'Please complete basic details first.',
                'error': 'basic_details_incomplete',
                'worker': 'none',
                'next_action': 'collect_details'
            }), 400
        
        # Check if KYC is complete
        if current_state['missing_kyc_fields']:
            return jsonify({
                'message': 'Please complete KYC details first.',
                'error': 'kyc_incomplete',
                'worker': 'none',
                'next_action': 'collect_kyc'
            }), 400
        
        # Perform fraud check
        fraud_result = fraud_agent.perform_fraud_check(current_state['entities'])
        
        # Update master agent
        user_master_agent.set_fraud_result(
            fraud_score=fraud_result['fraud_score'],
            fraud_flag=fraud_result['fraud_flag']
        )
        
        # Store in session
        session['fraud_result'] = fraud_result
        session['last_state'] = user_master_agent.state.copy()
        
        response = {
            'fraud_check': fraud_result,
            'session_id': session_id,
            'passed': fraud_result['fraud_flag'] != 'High'
        }
        
        if fraud_result['fraud_flag'] == 'High':
            response['message'] = '❌ Fraud check failed. Application cannot proceed.'
            response['worker'] = 'none'
            response['next_action'] = 'terminate'
            try:
                log_chat_event(session_id, 'status_change', {
                    'role': 'system',
                    'text': None,
                    'status': 'rejected',
                    'details': fraud_result
                })
            except Exception:
                pass
        else:
            response['message'] = '✅ Fraud check passed. Proceeding to underwriting...'
            response['worker'] = 'underwriting'
            response['action'] = 'call_underwriting_api'
            try:
                log_chat_event(session_id, 'action', {
                    'role': 'system',
                    'text': 'fraud_check_passed',
                    'status': 'approved',
                    'details': fraud_result
                })
            except Exception:
                pass
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in /fraud: {e}")
        return jsonify({
            'error': 'fraud_check_failed',
            'message': 'Fraud detection failed.'
        }), 500

@app.route('/documentation', methods=['POST'])
def documentation():
    """
    Final step: Generate sanction letter.
    """
    try:
        session_id = request.headers.get('X-Session-ID')
        
        if not session_id or session_id not in user_sessions:
            return jsonify({'error': 'Invalid or expired session.'}), 400
        
        session = user_sessions[session_id]
        user_master_agent = session['master_agent']
        current_state = user_master_agent.state
        
        # Verify conditions
        if not current_state.get('offer_accepted', False):
            return jsonify({
                'error': 'offer_not_accepted',
                'message': 'Please accept the offer first.'
            }), 400
        
        entities = current_state.get('entities', {})
        if not all(entities.get(field) for field in ['pan', 'aadhaar', 'address']):
            return jsonify({
                'error': 'kyc_incomplete',
                'message': 'KYC details are incomplete.'
            }), 400
        
        # Generate sanction letter (text + PDF)
        letter_data = generate_sanction_letter_text(current_state)
        pdf_path = generate_sanction_pdf(current_state)
        
        # Update final state
        user_master_agent.state['stage'] = ConversationStage.CLOSED
        user_master_agent.state['sanction_letter'] = letter_data['metadata']['sanction_id']
        user_master_agent.state['letter_generated_at'] = datetime.now().isoformat()
        user_master_agent.state['workflow_progress'] = 100
        
        # Update session
        session['sanction_letter'] = letter_data
        session['sanction_letter_pdf'] = pdf_path
        session['last_state'] = user_master_agent.state.copy()
        session['completed_at'] = datetime.now().isoformat()

        resp = jsonify({
            'message': '✅ Sanction letter generated successfully!',
            'letter_content': letter_data['content'],
            'metadata': letter_data['metadata'],
            'session_id': session_id,
            'stage': 'completed',
            'workflow_progress': 100,
            'download_url': f'/download/{session_id}',
            'next_action': 'download_letter'
        })

        resp.headers['X-Session-ID'] = session_id
        try:
            log_chat_event(session_id, 'action', {
                'role': 'system',
                'text': 'sanction_letter_generated',
                'status': 'approved',
                'details': letter_data['metadata']
            })
        except Exception:
            pass
        return resp
        
    except Exception as e:
        print(f"Error in /documentation: {e}")
        return jsonify({
            'error': 'documentation_failed',
            'message': 'Failed to generate sanction letter.'
        }), 500

@app.route('/download/<session_id>', methods=['GET'])
def download_sanction_letter(session_id):
    """Download the generated sanction letter PDF for a session."""
    session = user_sessions.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    pdf_path = session.get('sanction_letter_pdf')
    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({'error': 'No sanction letter available for download'}), 404

    directory, filename = os.path.split(pdf_path)
    return send_from_directory(directory, filename, as_attachment=True)

@app.route('/workflow/status', methods=['GET'])
def workflow_status():
    """Get current workflow status."""
    session_id = request.headers.get('X-Session-ID')
    
    if not session_id or session_id not in user_sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session = user_sessions[session_id]
    user_master_agent = session['master_agent']
    
    # Get workflow status from master agent
    workflow_status = user_master_agent.get_workflow_status()
    
    return jsonify({
        "session_id": session_id,
        "current_stage": workflow_status["current_stage"],
        "progress": workflow_status["progress"],
        "completed_stages": workflow_status["completed_stages"],
        "missing_fields": workflow_status["missing_fields"],
        "missing_kyc_fields": workflow_status["missing_kyc_fields"],
        "entities_collected": {k: v for k, v in user_master_agent.state["entities"].items() if v},
        "next_worker": determine_worker_from_stage(user_master_agent.state["stage"])
    })

@app.route('/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get session status (for debugging)."""
    if session_id in user_sessions:
        session = user_sessions[session_id]
        # Don't expose full agent state, just summary
        return jsonify({
            'session_id': session_id,
            'created_at': session.get('created_at'),
            'last_activity': session.get('last_activity'),
            'interaction_count': session.get('interaction_count', 0),
            'current_stage': session.get('master_agent').state.get('stage').value,
            'workflow_progress': session.get('master_agent').state.get('workflow_progress', 0),
            'has_offer': session.get('master_agent').state.get('offer_accepted', False)
        })
    return jsonify({'error': 'Session not found'}), 404

@app.route('/reset/<session_id>', methods=['POST'])
def reset_session(session_id):
    """Reset a session."""
    if session_id in user_sessions:
        user_sessions[session_id] = {
            'master_agent': MasterAgent(),
            'last_activity': time.time(),
            'created_at': datetime.now().isoformat(),
            'interaction_count': 0,
            'workflow_history': []
        }
        return jsonify({'message': 'Session reset successfully.'})
    return jsonify({'error': 'Session not found'}), 404

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(user_sessions),
        'llm_provider': llm_provider,
        'llm_mode': os.getenv("LLM_MODE", "enabled"),
        'agents': ['master', 'underwriting', 'sales', 'fraud']
    })

# --- File uploads serving (for admin) ---
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- Admin helpers ---
def is_admin_logged_in():
    return session.get('admin_username') is not None

@app.route('/bank/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        admins = read_csv(ADMINS_CSV)
        for adm in admins:
            if adm.get('username') == username and adm.get('password') == password:
                session['admin_username'] = username
                return redirect(url_for('admin_dashboard'))
        return render_template('admin_login.html', error='Invalid credentials')
    return render_template('admin_login.html')

@app.route('/bank/admin/logout')
def admin_logout():
    session.clear()
    return redirect(url_for('admin_login'))

@app.route('/bank/admin/dashboard')
def admin_dashboard():
    if not is_admin_logged_in():
        return redirect(url_for('admin_login'))
    applications = read_csv(APPLICATIONS_CSV)
    return render_template('admin_dashboard.html', applications=applications)

@app.route('/bank/admin/update_application', methods=['POST'])
def update_application():
    if not is_admin_logged_in():
        return redirect(url_for('admin_login'))

    session_id = request.form.get('session_id')
    timestamp = request.form.get('timestamp')
    new_status = request.form.get('status')
    reason = request.form.get('rejection_reason', '')

    rows = read_csv(APPLICATIONS_CSV)
    if rows:
        headers = list(rows[0].keys())
    else:
        headers = [
            'timestamp', 'session_id', 'full_name', 'phone', 'email',
            'city', 'loan_amount', 'status', 'rejection_reason', 'attached_files'
        ]

    for r in rows:
        if r.get('session_id') == session_id and r.get('timestamp') == timestamp:
            r['status'] = new_status
            r['rejection_reason'] = reason
            break

    ensure_csv(APPLICATIONS_CSV, headers)
    import csv as _csv
    with open(APPLICATIONS_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = _csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    log_chat_event(session_id, 'status_change', {
        'role': 'admin',
        'text': None,
        'status': new_status,
        'details': {'rejection_reason': reason},
    })

    return redirect(url_for('admin_dashboard'))

@app.route('/bank/admin/tune', methods=['GET', 'POST'])
def admin_tune():
    if not is_admin_logged_in():
        return redirect(url_for('admin_login'))

    headers = ['timestamp', 'admin', 'type', 'content', 'filename']
    if request.method == 'POST':
        description = request.form.get('description', '').strip()
        upload = request.files.get('file')
        filename = ''
        if upload and upload.filename:
            filename = secure_filename(upload.filename)
            upload.save(os.path.join(UPLOADS_DIR, filename))

        row = {
            'timestamp': datetime.utcnow().isoformat(),
            'admin': session.get('admin_username'),
            'type': 'text_and_file',
            'content': description,
            'filename': filename,
        }
        append_csv(TUNING_CSV, row, headers)

        return redirect(url_for('admin_tune'))

    items = read_csv(TUNING_CSV)
    return render_template('admin_tune.html', items=items)

@app.route('/bank/admin/tune/clear', methods=['POST'])
def admin_tune_clear():
    if not is_admin_logged_in():
        return redirect(url_for('admin_login'))

    headers = ['timestamp', 'admin', 'type', 'content', 'filename']
    # Clear file by writing just headers
    with open(TUNING_CSV, mode='w', newline='', encoding='utf-8') as f:
        import csv as _csv
        writer = _csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

    return redirect(url_for('admin_tune'))

@app.route('/bank/admin/logs')
def admin_logs():
    if not is_admin_logged_in():
        return redirect(url_for('admin_login'))
    logs = read_csv(CHAT_LOGS_CSV)
    return render_template('admin_logs.html', logs=logs)

# --- Application submission (for dashboard) ---
@app.route('/api/widget/submit_application', methods=['POST'])
def submit_application():
    session_id = request.headers.get('X-Session-ID') or f"session_{uuid.uuid4().hex[:16]}"
    data = request.get_json(silent=True) or {}
    if not data:
        return jsonify({'error': 'Invalid request'}), 400

    headers = [
        'timestamp', 'session_id', 'full_name', 'phone', 'email',
        'city', 'loan_amount', 'status', 'rejection_reason', 'attached_files',
    ]
    row = {
        'timestamp': datetime.utcnow().isoformat(),
        'session_id': session_id,
        'full_name': data.get('full_name', ''),
        'phone': data.get('phone', ''),
        'email': data.get('email', ''),
        'city': data.get('city', ''),
        'loan_amount': data.get('loan_amount', ''),
        'status': 'under_review',
        'rejection_reason': '',
        'attached_files': json.dumps(data.get('files', []), ensure_ascii=False),
    }
    append_csv(APPLICATIONS_CSV, row, headers)

    log_chat_event(session_id, 'action', {
        'role': 'user',
        'text': 'submit_application',
        'status': 'under_review',
        'details': row,
    })

    return jsonify({'ok': True, 'session_id': session_id})

# --- File upload endpoint for widget/applications ---
@app.route('/api/widget/upload', methods=['POST'])
def widget_upload():
    session_id = request.headers.get('X-Session-ID') or f"session_{uuid.uuid4().hex[:16]}"
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(f"{session_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}")
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        log_chat_event(session_id, 'action', {
            'role': 'user',
            'text': 'upload_file',
            'status': None,
            'details': {'filename': filename},
        })
        return jsonify({'filename': filename, 'url': url_for('uploaded_file', filename=filename, _external=False), 'session_id': session_id})

    return jsonify({'error': 'Invalid file type'}), 400

@app.template_filter('load_json')
def load_json_filter(s):
    try:
        return json.loads(s)
    except Exception:
        return []

if __name__ == '__main__':
    print("=" * 60)
    print("CREDGEN Loan Application System")
    print("=" * 60)
    print(f"Server starting on http://0.0.0.0:5000")
    print(f"LLM Provider: {llm_provider}")
    print(f"LLM Mode: {os.getenv('LLM_MODE', 'enabled')}")
    print("Available endpoints:")
    print("  GET  /                     - Frontend page (index.html)")
    print("  POST /chat                 - Main conversation endpoint")
    print("  POST /underwrite           - Underwriting process")
    print("  POST /sales                - Sales and negotiation")
    print("  POST /fraud                - Fraud detection")
    print("  POST /documentation        - Generate sanction letter")
    print("  GET  /workflow/status      - Check workflow progress")
    print("  GET  /health               - Health check")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
