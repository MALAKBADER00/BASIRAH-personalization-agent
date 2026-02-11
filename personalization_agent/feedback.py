import streamlit as st
from config import OPENAI_MODEL, OPENAI_API_KEY
from langchain_core.output_parsers import JsonOutputParser
from openai import OpenAI
import textwrap
import plotly.graph_objects as go

# Initialize LLM client
'''if "llm" not in st.session_state:
    st.session_state.llm = OpenAI() '''

# Trust Thresholds & Categories (same refactor as feedback (2).py)
TRUST_THRESHOLDS = {
    "no_info": (0, 3.9999),
    "basic_info": (4, 6.9999),
    "sensitive_info": (7, 8.9999),
    "critical_info": (9, 10.0000)
}

INFO_CATEGORIES = {
    "basic": ["name", "location", "job"],
    "sensitive": ["phone", "email", "address"],
    "critical": ["password", "ssn", "account", "credit_card"]
}

# ---------- FEEDBACK AGENT ----------
class FeedbackAgent:
    def __init__(self, conversation_results: list, llm):
        self.results = conversation_results
        self.llm = llm
        self.metrics = {}
        self.score = 0
        self.feedback_text = {}
        self.voice_feedback = ""

    # ---------- LOGIC ANALYSIS ----------
    def analyze_triggers(self):
        trigger_count = sum(len(r.get("detected_triggers", [])) for r in self.results)
        trigger_repetition = trigger_count / max(1, len(self.results))
        return trigger_count, trigger_repetition

    def analyze_trust_trends(self):
        trust_scores = [r.get("trust_score", 0) for r in self.results]
        increases = sum(1 for i in range(1, len(trust_scores)) if trust_scores[i] > trust_scores[i-1])
        decreases = sum(1 for i in range(1, len(trust_scores)) if trust_scores[i] < trust_scores[i-1])
        return increases, decreases

    def analyze_info_ratio(self):
        total_msgs = len(self.results)
        total_info = sum(len(r.get("info_to_reveal", [])) for r in self.results)
        ratio = total_info / max(1, total_msgs)
        return total_info, ratio

    def analyze_mistakes(self):
        mistakes = [
            log for r in self.results
            for log in r.get("analysis_log", [])
            if "BREACH" in log
        ]
        return len(mistakes)

    def analyze_phases(self):
        trust_scores = [r.get("trust_score", 0) for r in self.results]
        if not trust_scores:
            return "neutral"
        return "increment" if trust_scores[-1] > trust_scores[0] else "decrement"

    def compute_metrics(self):
        triggers, repetition = self.analyze_triggers()
        inc, dec = self.analyze_trust_trends()
        total_info, ratio = self.analyze_info_ratio()
        mistakes = self.analyze_mistakes()
        phases = self.analyze_phases()

        self.metrics = {
            "trigger_count": triggers,
            "trigger_repetition": repetition,
            "trust_increases": inc,
            "trust_decreases": dec,
            "info_revealed": total_info,
            "info_ratio": ratio,
            "mistakes": mistakes,
            "phase_trend": phases
        }

    def calculate_score(self):
        score = 10
        score -= self.metrics["mistakes"] * 1.5
        score -= self.metrics["info_ratio"] * 2
        score += min(self.metrics["trigger_count"], 5) * 0.5
        score += self.metrics["trust_increases"] * 0.2
        self.score = max(0, min(10, round(score, 1)))

    # ---------- AI FEEDBACK (PERSONALIZATION KEPT) ----------
    def generate_ai_feedback(self):
        prompt = f"""
        You are a **phishing training coach** specializing in the **Personalization Principle**.
        Your tone must be direct, constructive, and written in the second person ("you").

        DO NOT speak in third person.
        Evaluate how effectively the trainee personalized the interaction.

        Focus on:
        - How well you adapted language to the target
        - Whether you referenced personal/contextual details convincingly
        - Whether personalization increased trust or caused suspicion

        Metrics:
        {self.metrics}

        Conversation Log:
        {self.results}

        Output JSON only:
        {{
            "strengths": ["What you did well"],
            "weaknesses": ["Where personalization failed or was weak"],
            "turn_analysis": {{ "Turn 1": "...", "Turn 2": "..." }},
            "suggestions": ["Actionable improvements"]
        }}
        """
        parser = JsonOutputParser()
        response = self.llm.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        self.feedback_text = parser.parse(response.choices[0].message.content)

    def generate_ai_voice_feedback(self):
        prompt = f"""
        You are a phishing training coach speaking directly to a trainee.
        Give short, informal voice-style feedback about their use of **Personalization**.

        Style: mentor-like, conversational, concise.

        Data:
        - Score: {self.score}/10
        - Mistakes: {self.metrics['mistakes']}
        - Triggers used: {self.metrics['trigger_count']}

        Highlight one strong personalization moment, one mistake, and one improvement.
        Output a single paragraph suitable for speech.
        """
        response = self.llm.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        self.voice_feedback = response.choices[0].message.content
        return self.voice_feedback

    # ---------- RUN ----------
    def run(self):
        self.compute_metrics()
        self.calculate_score()
        self.generate_ai_feedback()
        return {
            "score": self.score,
            "metrics": self.metrics,
            "feedback": self.feedback_text
        }
