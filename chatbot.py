"""
chatbot.py
Conversational agent with multi-turn memory, RAG retrieval, and
multi-metric reasoning injection.
"""

import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from knowledge_base import retrieve
from reasoning_engine import analyse_inputs, format_analysis_for_prompt

load_dotenv()

# ---------------------------------------------------------------------------
# System prompt – defines the AI environmental scientist persona
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are Darukaa, an AI environmental scientist specialising in biodiversity intelligence.

RULES:
1. Every recommendation must state: WHAT to do, WHY it works (mechanism), WHICH metric improves, a CREDIBLE SOURCE, and a TIME HORIZON (short <1yr / medium 1–5yr / long >5yr).
2. Always reason across ≥ 3 environmental variables simultaneously. Never give single-variable answers.
3. If the user's input lacks key variables (soil, rainfall, land use, temperature, species data), ask targeted clarifying questions before recommending.
4. Structure every recommendation response as:
   ─────────────────────────────────────────
   RECOMMENDATION [N]: <title>
   Action      : <specific action>
   Mechanism   : <scientific explanation>
   Metrics     : <which variables improve and by how much>
   Time Horizon: <short/medium/long>
   Source      : <citation>
   ─────────────────────────────────────────
5. After recommendations, add a CONFIDENCE note (High/Medium/Low) with reasoning.
6. Never give vague advice like "use sustainable practices". Be quantitative.
7. Use the KNOWLEDGE BASE CONTEXT and METRICS ANALYSIS provided in each message.
"""

# ---------------------------------------------------------------------------
# Clarifying questions map – which variables to ask about if missing
# ---------------------------------------------------------------------------
CLARIFYING_QUESTIONS = {
    "soil_organic_carbon_pct": "What is the soil organic carbon (%) of the land?",
    "soil_ph":                 "What is the soil pH?",
    "rainfall_mm_year":        "What is the approximate annual rainfall (mm/year)?",
    "temperature_c":           "What is the mean annual temperature (°C)?",
    "land_use":                "What is the current land use? (e.g., monoculture, agroforestry, degraded forest, urban)",
}


class DarukaaChatbot:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        self._history = ChatMessageHistory()
        self._pending_data: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _build_messages(self, user_text: str, rag_chunks: list, metrics_analysis: str):
        history = self._history.messages[-20:]  # last 10 turns = 20 messages

        context_block = ""
        if rag_chunks:
            context_block += "\n\n=== KNOWLEDGE BASE CONTEXT ===\n" + "\n\n".join(
                f"[{i+1}] {chunk}" for i, chunk in enumerate(rag_chunks)
            )
        if metrics_analysis:
            context_block += "\n\n" + metrics_analysis

        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in history:
            messages.append(msg)
        messages.append(HumanMessage(content=user_text + context_block))
        return messages

    # ------------------------------------------------------------------
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Try to parse JSON from user message."""
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return None

    # ------------------------------------------------------------------
    def chat(self, user_input: str) -> str:
        # 1. Try to extract structured JSON data from the message
        json_data = self._extract_json(user_input)
        if json_data:
            self._pending_data.update(json_data)

        # 2. Run metrics analysis on accumulated data
        analysis = analyse_inputs(self._pending_data)
        metrics_str = format_analysis_for_prompt(analysis) if self._pending_data else ""

        # 3. If critical variables are missing and user seems to be asking for recommendations
        recommendation_keywords = {"recommend", "suggest", "improve", "help", "declining", "bad", "problem", "fix", "what should"}
        is_asking_for_help = any(kw in user_input.lower() for kw in recommendation_keywords)
        has_enough_data = len(self._pending_data) >= 3

        if is_asking_for_help and analysis["missing"] and not has_enough_data:
            missing_qs = [CLARIFYING_QUESTIONS[m] for m in analysis["missing"] if m in CLARIFYING_QUESTIONS]
            if missing_qs:
                clarification = (
                    "To give you precise, evidence-backed recommendations I need a few more details:\n\n"
                    + "\n".join(f"• {q}" for q in missing_qs[:3])
                    + "\n\nYou can answer in plain text or send a JSON block like:\n"
                    '{"soil_organic_carbon_pct": 0.3, "rainfall_mm_year": 300, "land_use": "monoculture"}'
                )
                self._history.add_user_message(user_input)
                self._history.add_ai_message(clarification)
                return clarification

        # 4. RAG retrieval – combine user query + detected interactions for richer retrieval
        rag_query = user_input + " " + " ".join(analysis["interactions"])
        rag_chunks = retrieve(rag_query, n_results=4)

        # 5. Build messages and call LLM
        messages = self._build_messages(user_input, rag_chunks, metrics_str)
        response = self.llm.invoke(messages)
        answer = response.content if isinstance(response.content, str) else response.content[0].get("text", str(response.content))

        # 6. Save to memory
        self._history.add_user_message(user_input)
        self._history.add_ai_message(answer)
        return answer

    # ------------------------------------------------------------------
    def reset(self):
        self._history.clear()
        self._pending_data = {}
