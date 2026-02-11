import pandas as pd
import re
from typing import Dict, List, Any
import logging
from config import VICTIM_CONFIG, TRUST_THRESHOLDS, INFO_CATEGORIES, OPENAI_MODEL, OPENAI_API_KEY
logger = logging.getLogger(__name__)
import json
from openai import OpenAI
import config
from langchain_core.output_parsers import JsonOutputParser
from dataclasses import dataclass
import random






class TriggerAnalyzer:
    def __init__(self, trigger_file: str = "trigger_words.xlsx"):
        self.df = pd.read_excel(trigger_file)

        # Initialize OpenAI client with API key from config
        self.openai = OpenAI(api_key=config.OPENAI_API_KEY)

        # Prepare few-shot examples from Excel to include in prompt
        self.examples = []
        for _, row in self.df.iterrows():
            example = {
                "trigger": str(row["Specific trigger words"]).strip(),
                "score": int(row["score (1-10)"]),
                "keywords": str(row["Keywords"]).strip(),
                "victim_reaction": str(row.get("victim reactions to these trigger words", "")).strip()
            }
            self.examples.append(example)
        
        self.parser = JsonOutputParser()


    def analyze(self, user_input: str) -> Dict[str, Any]:
        user_input_clean = ' '.join(user_input.strip().split())
        prompt = self._build_prompt(user_input_clean)

        try:
            response = self.openai.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a personalization phrase detector."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=300,
            )
            print(response)
            result_text = response.choices[0].message.content.strip()

            # Expecting a JSON array in response, parse it
            detected_triggers = self.parser.parse(result_text)

            # Ensure structure matches expected dict list
            cleaned = []
            for trig in detected_triggers:
                if isinstance(trig, dict) and "trigger" in trig and "score" in trig:
                    cleaned.append({
                        "trigger": str(trig["trigger"]),
                        "keywords": "",
                        "score": int(trig["score"]),
                        "expected_response": ""
                    })
            detected_triggers = cleaned

        except Exception as e:
            print(f"OpenAI error in TriggerAnalyzer: {e}")
            detected_triggers = self._legacy_analyze(user_input_clean)

        count = len(detected_triggers)
        total_score = min((t["score"] for t in detected_triggers), default=0)

        return {
            "triggers": detected_triggers,
            "count": count,
            "effectiveness": total_score,
            "total_score": total_score
        }

    def _build_prompt(self, user_input: str) -> str:
        examples_text = (
            "You are a Personalization Trigger Detector for scam detection.\n"
            "You analyze user sentences to find any phrase that creates trust, familiarity, or a sense of exclusivity by making the message feel tailored to the victim.\n"
            "For each detected phrase, give it a score from 1 to 10:\n"
            f"- 1–4: Mild personalization (e.g., 'Hello {config.info_data["name"]}', 'as you works as {config.info_data['job']}').\n"
            f"- 5–7: Medium personalization (e.g., 'We noticed you recently contacted us from {config.info_data['phone']}', 'Following up on the email sent to {config.info_data['email']}', 'We sent a package to {config.info_data['address']}, please confirm receipt').\n"
            f"- 8–10: Strong personalization or exploitation of identity/relationships (e.g., 'Your account number {config.info_data['account']} requires verification', 'We need your credit card details {config.info_data['credit_card']} to process the refund').\n\n"
            "You have the following known triggers with scores:\n"
        )

        # Few-shot examples from Excel
        for ex in self.examples[:8]:  # Only first 8 to avoid huge prompt
            examples_text += f"- \"{ex['trigger']}\" → Score: {ex['score']}\n"

        examples_text += (
            "\nTASK:\n"
            "1. Read the user sentence.\n"
            "2. Detect ALL personalization phrases — even if not in the known list.\n"
            "3. For each phrase, assign a score using the known list as reference.\n"
            "4. Output ONLY JSON (no code blocks, no text), in the format:\n"
            "[{\"trigger\": \"detected phrase\", \"score\": number}]\n"
            "If nothing is found, output []\n\n"
            f"User input: \"{user_input}\"\n"
        )
        return examples_text


    def _legacy_analyze(self, user_input: str) -> List[Dict[str, Any]]:
        # Original exact and keyword matching logic fallback

        detected = []
        total_score = 0

        user_lower = user_input.lower().strip()

        # STEP 1: Check if user speech CONTAINS any trigger phrases
        exact_matches = []
        for _, row in self.df.iterrows():
            trigger_phrase = str(row["Specific trigger words"]).lower().strip()
            # Remove ALL punctuation and extra spaces
            trigger_phrase = re.sub(r'[^\w\s]', '', trigger_phrase).strip()
            trigger_phrase = ' '.join(trigger_phrase.split())  # Remove extra spaces

            # Clean user input the same way
            clean_user_input = re.sub(r'[^\w\s]', '', user_lower).strip()
            clean_user_input = ' '.join(clean_user_input.split())

            # Check if trigger phrase is contained in user speech
            if trigger_phrase in clean_user_input:
                exact_matches.append({
                    "trigger": row["Specific trigger words"],
                    "keywords": row["Keywords"],
                    "score": int(row["score (1-10)"]),
                    "expected_response": row.get("victim reactions to these trigger words", "")
                })

        if exact_matches:
            detected = exact_matches
            total_score = min(trigger["score"] for trigger in detected) if detected else 0
        else:
            # STEP 2: Check keywords and find MINIMUM score for each keyword
            keyword_scores = {}  # keyword -> minimum score
            keyword_data = {}    # keyword -> row data with minimum score

            for _, row in self.df.iterrows():
                keywords = str(row["Keywords"]).lower().strip()
                score = int(row["score (1-10)"])

                # Check if this keyword is in user input
                if keywords in user_lower:
                    if keywords not in keyword_scores or score < keyword_scores[keywords]:
                        keyword_scores[keywords] = score
                        keyword_data[keywords] = {
                            "trigger": row["Specific trigger words"],
                            "keywords": row["Keywords"],
                            "score": score,
                            "expected_response": row.get("victim reactions to these trigger words", "")
                        }

            for keyword, data in keyword_data.items():
                detected.append(data)

            total_score = min(trigger["score"] for trigger in detected) if detected else 0

        return detected
    

class TrustCalculator:
    def __init__(self):
        self.config = VICTIM_CONFIG
        self.openai = OpenAI(api_key=config.OPENAI_API_KEY)
        self.parser = JsonOutputParser()

    def calculate_trigger_impact(self):
        pass

    def detect_inappropriate_requests(seld):
        pass

    def analyze_context_with_ai(self):
        pass
        
    def calculate_total_change(self) -> Dict[str, Any]:
        """Calculate new trust score"""
        
        return {
            "new_trust": new_trust,
            "change": trust_increase,
            "reasoning": f"Trust increased by {trust_increase:.2f} due to {len(triggers)} effective triggers"
        }
    

class VulnerabilityAssessor:
    def __init__(self):
        self.thresholds = TRUST_THRESHOLDS
        self.info_categories = INFO_CATEGORIES
        
    def assess(self, trust_score: float, user_input: str) -> Dict[str, Any]:
        """Assess what information to reveal"""
        # Extract info requests
        requested_info = self._extract_requests(user_input)
        
        # ✅ FIX: Only reveal if information was specifically requested
        info_to_reveal = []
        category_revealed = "none"
        
        # Determine what CAN be revealed based on trust level
        available_info = []
        if trust_score >= self.thresholds["critical_info"][0]:
            available_info.extend(self.info_categories["critical"])
            available_info.extend(self.info_categories["sensitive"])
            available_info.extend(self.info_categories["basic"])
            category_revealed = "critical"
        elif trust_score >= self.thresholds["sensitive_info"][0]:
            available_info.extend(self.info_categories["sensitive"])
            available_info.extend(self.info_categories["basic"]) 
            category_revealed = "sensitive"
        elif trust_score >= self.thresholds["basic_info"][0]:
            available_info.extend(self.info_categories["basic"])
            category_revealed = "basic"
        
        # ✅ KEY FIX: Only reveal what was BOTH requested AND available
        if requested_info:
            info_to_reveal = [info for info in requested_info if info in available_info]
            # Update category based on what's actually being revealed
            if any(info in self.info_categories["critical"] for info in info_to_reveal):
                category_revealed = "critical"
            elif any(info in self.info_categories["sensitive"] for info in info_to_reveal):
                category_revealed = "sensitive"
            elif any(info in self.info_categories["basic"] for info in info_to_reveal):
                category_revealed = "basic"
            else:
                category_revealed = "none"
        else:    
            # ✅ If nothing was requested, reveal nothing
            info_to_reveal = []
            category_revealed = "none"
        
        return {
            "should_reveal": len(info_to_reveal) > 0,
            "info_to_reveal": info_to_reveal,
            "category": category_revealed,
            "vulnerability_level": trust_score / 10,
            "trust_threshold": self._get_trust_level_name(trust_score)
        }



    def _get_trust_level_name(self, trust_score: float) -> str:
        """Get trust level name based on score"""
        if trust_score >= self.thresholds["critical_info"][0]:
            return "CRITICAL"
        elif trust_score >= self.thresholds["sensitive_info"][0]:
            return "SENSITIVE"
        elif trust_score >= self.thresholds["basic_info"][0]:
            return "BASIC"
        else:
            return "NO ACCESS"
    
    def _extract_requests(self, user_input: str) -> List[str]:
        """Extract information requests"""
        requests = []
        user_lower = user_input.lower()
        
        keywords = {
        "name": ["name", "who are you", "full name", "what's your name"],
        "phone": ["phone", "number", "call", "mobile", "telephone"],
        "email": ["email", "@", "contact", "mail", "e-mail"],
        "location": ["location", "where are you", "where do you live", "city", "country", "occupation"],  # ✅ Added separate location
        "job": ["job", "work", "working", "occupation", "where do you work", "what do you do"],  # ✅ Added job detection
        "address": ["address", "home address", "where", "live", "street", "building"],  # ✅ Separated from location
        "password": ["password", "login", "pin", "code", "passcode"],
        "ssn": ["social security", "ssn", "social", "civil id", "id number"],  # ✅ Added civil id
        "account": ["account", "username", "user id", "account number"],
        "credit_card": ["card", "credit", "payment", "billing", "credit card"]  # ✅ Added credit card
    }
            
        for info_type, words in keywords.items():
            if any(word in user_lower for word in words):
                requests.append(info_type)
                
        return requests