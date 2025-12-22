import os
import logging
import json
import google.generativeai as genai
from typing import Dict, List, Optional, Any

# Set up logging
logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = None
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash') # using a fast model
                logger.info("Gemini Service initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
        else:
            logger.warning("No GEMINI_API_KEY found. Gemini Service will not work.")

    def is_available(self) -> bool:
        return self.model is not None

    def generate_response(
        self, 
        user_message: str, 
        system_prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generates a response from Gemini, mapped to the backend schema.
        
        Schema mapping:
        {
            "message": <gemini_response>,
            "worker": "none",
            "intent": "general_response", # or inferred
            "stage": "unknown", # preserve existing if possible
            "terminate": False
        }
        """
        if not self.model:
            return {
                "message": "Gemini is not configured.",
                "worker": "none",
                "intent": "error",
                "terminate": False
            }

        try:
            # Construct prompt contents
            # We explicitly prepend the system prompt as instructed since the API might treat 'system' role differently 
            # or we simple allow the model to handle it if we use the system_instruction.
            # However, prompt requirements said: "Precede all user input." and "Gemini client must accept systemPrompt... Construct payload..."
            
            # specific structure including 'suggestions'.
            
            prompt = f"""{system_prompt}

You must respond in strict JSON format.
Output schema:
{{
  "response": "Your natural language response here (can include markdown)",
  "suggestions": ["Short follow-up option 1", "Short follow-up option 2", "Option 3"]
}}

User Query: {user_message}
"""
            
            response = self.model.generate_content(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=500,
                    temperature=0.7,
                    response_mime_type="application/json"
                )
            )
            
            import json
            try:
                data = json.loads(response.text)
                return {
                    "message": data.get("response", ""),
                    "suggestions": data.get("suggestions", []),
                    "worker": "none",
                    "intent": "gemini_response",
                    "status": "success"
                }
            except json.JSONDecodeError:
                # Fallback if model fails to output JSON (rare with response_mime_type set)
                logger.warning("Gemini failed to return valid JSON, falling back to raw text.")
                return {
                    "message": response.text,
                    "suggestions": [],
                    "worker": "none",
                    "intent": "gemini_response",
                    "status": "success_text_only"
                }
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return {
                "message": "I'm having trouble connecting to my AI services right now.",
                "worker": "none",
                "intent": "error",
                "error": str(e)
            }
