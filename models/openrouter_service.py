import os
import json
import logging
import requests
from typing import Dict, List, Optional, Any

# Set up logging
logger = logging.getLogger(__name__)

class OpenRouterService:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model_name = "google/gemma-3-27b-it:free"
        
        if self.api_key:
            logger.info("OpenRouter Service initialized successfully.")
        else:
            logger.warning("No OPENROUTER_API_KEY found. Agent will likely fail.")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate_response(
        self, 
        user_message: str, 
        system_prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generates a response using OpenRouter with entity extraction.
        """
        if not self.api_key:
            return {
                "message": "AI Service not configured (Missing API Key).",
                "suggestions": [],
                "worker": "none",
                "intent": "error",
                "status": "error"
            }

        # Enhanced JSON instruction for entity extraction
        json_instruction = """
You must respond in strict JSON format. 
Your response should include natural language and extracted entities.

Output JSON schema:
{
  "response": "Your natural language response here (be conversational and helpful)",
  "suggestions": ["Short follow-up option 1", "Option 2", "Option 3"],
  "extracted_entities": {
    "loan_amount": null or number,
    "tenure": null or number (in months),
    "age": null or number,
    "income": null or number,
    "name": null or string,
    "employment_type": null or "salaried" or "self_employed" or "professional",
    "purpose": null or string,
    "pan": null or string (format: ABCDE1234F),
    "aadhaar": null or string (12 digits),
    "address": null or string,
    "pincode": null or string (6 digits)
  }
}

Entity Extraction Rules:
- loan_amount: Extract numbers with lakh/lac/L (e.g., '5 lakhs' = 500000, '₹5,00,000' = 500000)
- tenure: Extract years/months (e.g., '3 years' = 36, '24 months' = 24)
- income: Extract LPA/monthly (e.g., '8 LPA' = 800000, '50k per month' = 600000 annual)
- pan: 10 characters, format: 5 letters + 4 digits + 1 letter
- aadhaar: 12 digits, can have spaces/dashes
- pincode: 6 digits
- employment_type: 'salaried' for job/employee, 'self_employed' for business/self-employed
  """

        full_system_prompt = f"""{system_prompt}

{json_instruction}

IMPORTANT: Only extract entities that the user explicitly provides. If user doesn't mention something, keep it as null.
Be helpful and guide them through the loan application process step by step.
        """
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "CredGen AI Assistant"
        }
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": full_system_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
        
        # Add chat history if provided
        if chat_history:
            messages = [messages[0]] + chat_history + [messages[1]]
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(data),
                timeout=30 
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Clean the content (remove markdown code blocks if present)
                clean_content = content.strip()
                if "```json" in clean_content:
                    clean_content = clean_content.replace("```json", "").replace("```", "").strip()
                elif "```" in clean_content:
                    clean_content = clean_content.replace("```", "").strip()
                
                # Try to parse the JSON content
                try:
                    parsed = json.loads(clean_content)
                    
                    # Validate extracted entities
                    extracted_entities = parsed.get("extracted_entities", {})
                    validated_entities = {}
                    
                    # Only include non-null entities
                    for key, value in extracted_entities.items():
                        if value is not None:
                            # Basic validation
                            if key == "loan_amount" and isinstance(value, (int, float)) and value > 0:
                                validated_entities[key] = value
                            elif key == "tenure" and isinstance(value, (int, float)) and value > 0:
                                validated_entities[key] = int(value)
                            elif key == "age" and isinstance(value, (int, float)) and 18 <= value <= 80:
                                validated_entities[key] = int(value)
                            elif key == "income" and isinstance(value, (int, float)) and value > 0:
                                validated_entities[key] = value
                            elif key in ["name", "address", "purpose"] and isinstance(value, str) and value.strip():
                                validated_entities[key] = value.strip()
                            elif key == "employment_type" and value in ["salaried", "self_employed", "professional"]:
                                validated_entities[key] = value
                            elif key == "pan" and isinstance(value, str) and len(value.strip()) == 10:
                                validated_entities[key] = value.strip().upper()
                            elif key == "aadhaar" and isinstance(value, str):
                                # Remove spaces and dashes
                                clean_aadhaar = value.replace(" ", "").replace("-", "")
                                if clean_aadhaar.isdigit() and len(clean_aadhaar) == 12:
                                    validated_entities[key] = clean_aadhaar
                            elif key == "pincode" and isinstance(value, str) and value.isdigit() and len(value) == 6:
                                validated_entities[key] = value
                    
                    return {
                        "message": parsed.get("response", ""),
                        "suggestions": parsed.get("suggestions", []),
                        "extracted_entities": validated_entities,
                        "worker": "none",
                        "intent": "openrouter_response",
                        "status": "success"
                    }
                    
                except json.JSONDecodeError as e:
                    # Fallback if JSON parsing fails
                    logger.warning(f"OpenRouter returned invalid JSON: {e}\nContent: {clean_content[:200]}...")
                    return {
                        "message": clean_content,  # Return raw text
                        "suggestions": [],
                        "extracted_entities": {},
                        "worker": "none",
                        "intent": "openrouter_response",
                        "status": "success_text_only"
                    }
            else:
                logger.error(f"OpenRouter API Error: {response.status_code} - {response.text}")
                return {
                    "message": "I'm having trouble connecting to the AI system right now. Please try again.",
                    "intent": "error",
                    "worker": "none",
                    "status": "error"
                }
                
        except requests.exceptions.Timeout:
            logger.error("OpenRouter request timed out")
            return {
                "message": "The AI service is taking too long to respond. Please try again.",
                "intent": "error",
                "worker": "none",
                "status": "timeout"
            }
        except Exception as e:
            logger.error(f"OpenRouter Request Failed: {e}")
            return {
                "message": "AI service is currently unavailable.",
                "intent": "error",
                "worker": "none",
                "status": "error"
            }
