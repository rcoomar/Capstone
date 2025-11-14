# llm_config.py
import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

anthropic_llm = LLM(
    model="anthropic/claude-sonnet-4-5-20250929",  # ✅ correct canonical name
    api_key=ANTHROPIC_API_KEY,
    temperature=0.2
)
