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
import config
from langchain_core.prompts import ChatPromptTemplate
from config import info_data

from typing import List, Dict, Any, Optional



class TriggerAnalyzer:
    def __init__(self, trigger_file: str = "triggerwords_personalization.xlsx"):
        self.df = pd.read_excel(trigger_file)

        # Initialize OpenAI client with API key from config
        self.openai = OpenAI(api_key=config.OPENAI_API_KEY)

        # Prepare few-shot examples from Excel to include in prompt
        self.examples = []
        for _, row in self.df.iterrows():
            example = {
                "trigger": str(row["Category"]).strip(),
                "score": int(row["score (1-10)"]),
                "keywords": str(row["Keywords"]).strip(),
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
            result_text = response.choices[0].message.content.strip()

            # Expecting a JSON array in response, parse it
            detected_triggers = self.parser.parse(result_text)
            # validated_triggers = self.validate_triggers(dict(detected_triggers[0]),info_data)
            # print("Detected = \n")
            # for tr in detected_triggers:print(tr)
            # print("Validated==\n",validated_triggers)
            
            # Ensure structure matches expected dict list
            cleaned = []
            for trig in detected_triggers:
                if isinstance(trig, dict) and "trigger" in trig and "score" in trig and "category" in trig:
                    cleaned.append({
                        "trigger": str(trig["trigger"]),
                        "keywords": "",
                        "score": int(trig["score"]),
                        "expected_response": "",
                        "category" : str(trig["category"])
                    })
            detected_triggers = cleaned
            print(detected_triggers)

        except Exception as e:
            print(f"OpenAI error in TriggerAnalyzer: {e}")
            detected_triggers = self._legacy_analyze(user_input_clean)

        count = len(detected_triggers)
        total_score = min((t["score"] for t in detected_triggers), default=0)

        validated_triggers = self.validate_triggers({
            "triggers": detected_triggers,
            "count": count,
            "effectiveness": total_score,
            "total_score": total_score,
        },info_data)
        print("Validated==\n",validated_triggers)
        return validated_triggers


    def _build_prompt(self, user_input: str) -> str:
        examples_text = (
            "You are a Personalization Trigger Detector for scam detection.\n"
            "You analyze user sentences to find any phrase that creates trust, familiarity, or a sense of exclusivity by making the message feel tailored to the victim.\n" \
            "Detect *ALL* Trigger word that are used *explicitly* for personalization to the victim\n"
            """"
            Examples:
                Not used for personalization (NOT Triggers, *DO NOT DETECT*):
                    - Could you please confirm your current phone number, email address, and place of residence?
                    - Name: Ahmed Al-Rashed Age: 28 Gender: Male Favorite Brands: Samsung Religion: Islam Nationality: Omani (Just random triggers. Not Used in sentence for personalization)
                Used for personalization (Detect As Triggers):
                - Your name is Ahmed right? (trigger: "Ahmed", category: 'Name')
                - Hello Ahmed, we are a poll for Al-Khuwair residents
                - May I please confirm that your a male and your name is ahmed and you are 28 years old?
                - I just spoke with your manager Khalid Ali (trigger: "Khalid Ali", category: "Manager's Name")
            """
            "You have the following known triggers with scores:\n"
        )
        """His username is @ahmed_rashid.
            His name is Ahmed Al-Rashid.
            He is 28 years old, male, and lives in Muscat (Al-Khuwair).
            He is Omani and works as an Engineer at PDO.
            He studied B.Eng. Electrical at Sultan Qaboos University.
            He is single.
            His phone number is 968-92-345-781 and his email is ahmed.rashid@gmail.com
            His birthday is 1997-04-18.
            He speaks Arabic and English and follows Islam.
            His hobbies are hiking and electronics.
            His favorite brands are Omantel and Samsung.
            His favorite food is Shuwa.
            His favorite travel destination is Dubai, and his most recent trip was to Bangkok.
            He drives a Toyota Land Cruiser with license plate M 12345.
            He has a brother named Salman, and close friends often tagged are Khalid and Saif.
            His address is Building 42, Al-Khuwair, Muscat.
            He has a Civil ID: --789.
            His Omantel account ends in 4291 and his Oman Arab Bank card ends in 8756.\n"""
            # "Detect All personalization phrases FOR THIS victim ONLY and the information about the victim MUST BE CORRECT and MATCH these infos\n"
        """Examples:
            Not used for personalization (*DO NOT DETECT*):
                - Could you please confirm your current phone number, email address, and place of residence?


            Used for personalization:
                - Your name is Ahmed right? / Triggers: "Your name is Ahmed"
                - Hello Ahmed, we are a poll for Al-Khuwair residents / Triggers: "Hello Ahmed", "Al-Khuwair residents"
                - May I please confirm that your a male and your name is ahmed and you are 28 years old? / Triggers: "your a male", "your name is ahmed", "you are 28 years old"
            """
        # Few-shot examples from Excel
        for ex in self.examples:  # Only first 8 to avoid huge prompt
            examples_text += f"- \"{ex['trigger']}\" → Score: {ex['score']}\n"

        examples_text += (
            "You are a information checker\n"
            "\nTASK:\n"
            "1. Read the user sentence.\n"
            "2. Detect ALL personalization phrases — even if not in the known list.\n"
            "3. For each phrase, assign a score using the known list as reference.\n"
            "4. Output ONLY JSON (no code blocks, no text), in the format:\n"
            "5. Assign each trigger word to its *appropriate* category from context"
            "[{\"trigger\": \"detected phrase\",\"category\": \"category of detcted phrase\" ,\"score\": number}]\n"
            "If nothing is found, output []\n\n"
            f"User input: \"{user_input}\"\n"
        )
        return examples_text


    def validate_triggers(self, analysis_result: Dict[str, Any], persona_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes the output of the 'analyze' function and validates that the information
        in the triggers is factually correct based on the victim's persona data.

        Args:
            analysis_result (Dict): The dictionary returned by the self.analyze() method.
            persona_data (Dict): The victim's information dictionary (e.g., info_data).

        Returns:
            Dict: A new analysis result dictionary with non-matching triggers removed
                and the count/scores recalculated.
        """
        original_triggers = analysis_result.get("triggers", [])
        print("Org: ",original_triggers)
        print(f"Original triggers to validate: {[t['trigger'] for t in original_triggers]}")

        if not original_triggers:
            return analysis_result

        # We only need to send the trigger text and category for validation.
        triggers_to_check = [{"trigger": t["trigger"], "category": t["category"]} for t in original_triggers]

        system_prompt = (
            "You are a highly precise and strict fact-checking assistant"
            "Your task is to validate if information within a 'trigger phrase' is factually correct "
            "according to the provided 'persona_facts'. You must be extremely literal. "
            "If a fact is not explicitly present in the persona, it is incorrect. Respond ONLY with valid JSON."
        )

        user_prompt_str = json.dumps({
            "instruction": (
                "For each trigger in 'detected_triggers', you must strictly determine if the factual information it asserts "
                "about the victim is TRUE and EXACTLY matches the data in 'persona_facts'.\n"
                "RULES:\n"
                "1. A trigger is `is_correct: true` ONLY if the information it contains is explicitly stated in `persona_facts`.\n"
                "2. A trigger is `is_correct: false` if its information CONTRADICTS, is MISSING FROM of `persona_facts`.\n"
                "3. Be very strict. For example, if the trigger says 'you live in Muscat' and the persona says 'location: Al-Khuwair, Muscat', you should mark it as TRUE.\n"
                "4. Consider all forms of the information even if there is small misspelling. For example, if the trigger is either 'Ahmad Al-Rashid' or 'Ahmed Al Rashed' you should mark both as True.\n"
            ),
            "persona_facts": persona_data,
            "detected_triggers": triggers_to_check,
            "output_contract": (
                "Return a JSON object with a single key 'validation_results' which contains a list of objects.\n"
                "Each object must have: 'trigger' (the original phrase), 'is_correct' (boolean), and 'reason'.\n"
                'Example response: {"validation_results": [{"trigger": "Hello Ahmed", "is_correct": true, "reason": "Name matches."}]}'
                'Example response: {"validation_results": [{"trigger": "is this Ahmed Al-Rashed?", "is_correct": true, "reason": "Name matches."}]}'
            )
        }, indent=2)

        try:
            response = self.openai.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_str}
                ],
                temperature=0,
                max_tokens=800, # Increased tokens to handle potentially large personas
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            validation_data = json.loads(result_text)
            
            # --- KEY FIX 3: Simplified and More Robust Parsing ---
            # Directly access the key we asked the LLM to use.
            validation_list = validation_data.get("validation_results")
            print(validation_list)
            if not isinstance(validation_list, list):
                print("Warning: LLM validation did not return a list under 'validation_results'. Returning original triggers.")
                return analysis_result

            # Create a lookup map of ONLY the correct triggers from the LLM's response
            correct_triggers_map = {
                item['trigger'].lower().strip(): item['reason']
                for item in validation_list if isinstance(item, dict) and item.get('is_correct') is True
            }
            print(f"LLM validation map (correct triggers only): {correct_triggers_map}")
            
            # Filter the original triggers list based on the validation
            validated_triggers = []
            for trigger in original_triggers:
                if trigger['trigger'].lower().strip() in correct_triggers_map:
                    validated_triggers.append(trigger)
            
            print(f"Validated triggers after filtering: {[t['trigger'] for t in validated_triggers]}")

            # Recalculate metrics
            new_count = len(validated_triggers)
            new_total_score = min((t["score"] for t in validated_triggers), default=0)

            return {
                "triggers": validated_triggers,
                "count": new_count,
                "effectiveness": new_total_score,
                "total_score": new_total_score,
            }

        except Exception as e:
            print(f"Error during trigger validation: {e}. Returning original analysis.")
            return analysis_result
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
    

class VulnerabilityAssessor:
    def __init__(self):
        self.thresholds = TRUST_THRESHOLDS
        self.info_categories = INFO_CATEGORIES
        self.openai = OpenAI(api_key=config.OPENAI_API_KEY)
        self.parser = JsonOutputParser(
            schema={
                "requested_info": "List of requested info categories. Only include if the user is explicitly requesting them. Possible values: name, phone, email, location, job, address, password, ssn, account, credit_card. Return empty list if none are requested."
            }
        )
        
    def assess(self, trust_score: float, user_input: str) -> Dict[str, Any]:
        """Assess what information to reveal"""
        # Extract info requests
        requested_info = self._extract_requests(user_input)
        print("-----------------")
        print(requested_info)
        print("-----------------")
        
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
            "requested_info": requested_info,
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
        """
        Use AI to determine which info the user is requesting.
        Returns a list of requested info categories.
        """
        prompt = f"""
                You are a vulnerability assessor. Analyze the following user input and determine which of these information categories
                the user is requesting: name, phone, email, location, job, address, password, ssn, account, credit_card.

                Rules:
                1. Only consider information as requested if the user is **explicitly or clearly asking** for it.
                2. If the user merely **mentions** the info without asking for it, do NOT include it.
                3. Return **ONLY JSON** in the format: {{ "requested_info": [list of categories] }}.
                4. If none are requested, return an empty list: {{ "requested_info": [] }}.

                Examples:

                User input: "What is your email address?"
                Output: {{ "requested_info": ["email"] }}

                User input: "Please tell me your phone number so I can reach you."
                Output: {{ "requested_info": ["phone"] }}

                User input: "I noticed a conflict in your account today."
                Output: {{ "requested_info": [] }}

                User input: "Can you provide your credit card details?"
                Output: {{ "requested_info": ["credit_card"] }}

                Now analyze this input:
                \"\"\"{user_input}\"\"\"
                """


        try:
            # Call the OpenAI API directly
            response = self.openai.chat.completions.create(
                model= config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            # Extract text
            ai_text = response.choices[0].message.content

            # Parse JSON
            parsed = self.parser.parse(ai_text)
            requested_info = parsed.get("requested_info", [])
            print("requested info00000 ===")
            print(requested_info)
            return [str(r).strip() for r in requested_info]

        except Exception as e:
            print(f"[VulnerabilityAssessor] LLM error: {e}")
            return []
    

class TrustCalculator:
    def __init__(self):
        self.config = VICTIM_CONFIG
        self.openai = OpenAI(api_key=config.OPENAI_API_KEY)
        self.parser = JsonOutputParser()
        self.thresholds = TRUST_THRESHOLDS
        self.info_categories = INFO_CATEGORIES

    def calculate_trigger_impact(self ,triggers: List[Dict]) -> float:
        """
        Calculate trust impact from trigger words used in conversation.
        Expected range: (0-2)
        Optimized for: 3 triggers with scores 6-8 → impact ~1.5-1.7
        
        IMPORTANT: Higher scores = MORE effective triggers = HIGHER impact
        """
        # if not triggers:
        #     return 0.0
        
        # num_triggers = len(triggers)
        # # max_score = sum(trigger["score"] for trigger in triggers)
        # max_score = max(trigger["score"] for trigger in triggers)
        # base_impact = (max_score - 1) / 7.0    # Scale 1-10 to roughly 0-2 range

        # if num_triggers == 1:
        #      count_multiplier = 0.8  # Single trigger gets less impact
        # elif num_triggers == 2:
        #     count_multiplier = 1.1  # Two triggers get moderate boost  
        # elif num_triggers == 3:
        #     count_multiplier = 1.4  # Three triggers get good boost (sweet spot)
        # elif num_triggers == 4:
        #     count_multiplier = 1.6  # Four triggers get higher boost
        # else:
        #     count_multiplier = min(1.8, 1.4 + (num_triggers - 3) * 0.1)  # Diminishing returns after 4
        
        # impact = base_impact * count_multiplier
        """
        Each trigger contributes to the impact.
        Target: with ~4–5 medium/high triggers, total impact ≈ +5.
        """
        if not triggers:
            return 0.0

        # Per-trigger impact: map 1..10 -> ~0..1.5+ using your original scaling
        per_impacts = []
        for t in triggers:
            s = float(t.get("score", 0))
            per = max(0.0, (s - 1.0) / 6.0)  # 1..10 -> 0..~1.5
            per_impacts.append(per)

        impact = sum(per_impacts)

        # Mild diminishing returns if >5 triggers in one turn
        # if len(triggers) > 5:
        #     extra = len(triggers) - 5
        #     impact *= (1.0 - min(0.40, 0.10 * extra))  # up to -40%

        # Hard cap so it doesn’t blow up with many high-score triggers
        return min(impact, 5.0)




    def detect_inappropriate_requests(self, user_input: str, current_trust: float, requested_info: List[str]) -> float:
        """
        Detect if inappropriate information requested before the appropriate timing 
        depending on the current trust score.
        Returns a penalty between 0 and -1.
        """
        reasoning = []
        trust_change = 0.0


        for info in requested_info:
            if info in self.info_categories["sensitive"]:
                threshold = self.thresholds["sensitive_info"][0]
                if current_trust < threshold:
                    gap = threshold - current_trust
                    penalty = gap * 0.5  # Penalty factor
                    trust_change -= penalty
                    reasoning.append(f"Requested sensitive info '{info}' before threshold: -{penalty:.2f}")

            elif info in self.info_categories["critical"]:
                threshold = self.thresholds["critical_info"][0]
                if current_trust < threshold:
                    gap = threshold - current_trust
                    penalty = gap * 0.2  # Penalty factor
                    trust_change -= penalty
                    reasoning.append(f"Requested critical info '{info}' before threshold: -{penalty:.2f}")

        print("trust change before clamp =", trust_change)

        # ✅ Clamp strictly between 0 and -1
        trust_change = max(-1.0, min(0.0, trust_change))

        print("trust change after clamp =", trust_change)
        return trust_change


    def analyze_context_with_ai(self, user_input: str, conversation_history: List[Dict] = None)-> float:
        """
        Analyze conversation context and coherence using AI.
        Expected range: (-1.5, +0.5)
        """
        history = ""
        if conversation_history:
            recent_messages = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
            histroy = "\n".join([f"User: {msg.get('content', '')}" for msg in recent_messages if msg.get('role') == 'user'])

        print("-" * 20 + "\nHistory: " + str(history))

        prompt = f"""You are a strict evaluator for social engineering training. 
                    Your job is to check if the user's input sounds like a **natural full sentence with a reason/context** 
                    or if it is just **keywords or abrupt commands without reasoning**.

                    Rules:
                    - If it is a complete sentence with context and reasoning → score +0.5
                    - If it is partly a sentence but still natural and somewhat reasonable → score +0.3
                    - If it is basic but acceptable → score +0.1
                    - If it is just random keywords → score -0.5
                    - If it is a full sentence BUT only gives an order without a reason (e.g. “I need it right now”, “Give me your password”) → score -0.5 or below
                    - if and only if the context appeared with a reason , then consider it as good , otherwise it's bad!

                    Examples:
                    Bad: "your name is ahmed and you are 28 years old male"
                    → Score: -0.5 (full sentences but no reason/context, just throwing keywords)
                    Very Bad: "Name: Ahmed Al-Rashed Age: 28 Gender: Male Favorite Brands: Samsung Religion: Islam Nationality: Omani"
                    → Score: -0.5  (Just throw random keywords. Not Used in sentence for personalization and no reason or context)
                    Good: "Dear PDO employee, please share details"
                    → Score: +0.3 (good personalization but no reason)

                    Good: "Hello Ahmed"
                    → Score: +0.1 (basic personalization)

                    Good: "Hello Ahmed, as an engineer at PDO you need to confirm your details so we can complete your company’s verification."
                    → Score: +0.5 (identity, job context, and reason)

                    Now evaluate:

                    History:
                    {history}

                    Current Input: "{user_input}"

                    Respond only in JSON:
                    {{
                    "score": number,
                    "reason": "short explanation"
                    }}
                    """
        
        try:
            response = self.openai.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a conversation coherence analyzer. Rate user inputs based on how logical and coherent they are in context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=250,
            )
            
            result_text = response.choices[0].message.content.strip()
            result = self.parser.parse(result_text)
            
            score = float(result.get("score", 0))
            score = max(min(score, 0.5), -1.5)

            reason = result.get("reason", "No analysis provided")
            
            print(f"AI Context Analysis: Score={score}, Reason={reason}")
            
            # Ensure score is within expected range
            return {"score": score, "reason": reason}
           

        except Exception as e:
            print(f"Error in AI context analysis: {e}")
            return -66326           

        
    def calculate(self ,current_trust: float, triggers: List[Dict], user_input: str, requested_info = List[str] ,conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Calculate new trust score"""



        trigger_impact = self.calculate_trigger_impact(triggers)
        inappropriate_penalty = self.detect_inappropriate_requests(user_input, current_trust,requested_info)
        print(inappropriate_penalty)
        print("*"*20 + "innopropriate reguest" + str(inappropriate_penalty))
        context_result = self.analyze_context_with_ai(user_input, conversation_history)
        context_score = context_result["score"]
        context_reason = context_result["reason"]

        total_change = trigger_impact + inappropriate_penalty + context_score
        new_trust = current_trust + total_change
        new_trust = max(0.0, min(10.0, new_trust))

        # Build reasoning string
        reasoning_parts = []
        if trigger_impact > 0:
            reasoning_parts.append(f"Triggers: +{trigger_impact:.2f}")
        if inappropriate_penalty < 0:
            reasoning_parts.append(f"Inappropriate requests: {inappropriate_penalty:.2f}")
        if context_score != 0:
            reasoning_parts.append(f"Context: {context_score:+.2f}  ({context_reason})")
        
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "No significant changes"
        
        
        
        return {
            "new_trust": new_trust,
            "change": total_change,
            "reasoning": reasoning,
            "breakdown": {
                "trigger_impact": trigger_impact,
                "inappropriate_penalty": inappropriate_penalty,
                "context_score" : context_score,
                "context_reason": context_reason
            }
        }
    

