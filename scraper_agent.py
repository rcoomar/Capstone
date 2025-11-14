# src/mslinsightsv1/agents/scraper_agent.py
from crewai import Agent
from mslinsightsv1.agents.custom_selenium_tool import CustomSeleniumScraper
from mslinsightsv1.llm_config import anthropic_llm

custom_selenium_tool = CustomSeleniumScraper()

scraper_agent = Agent(
    role="Pharma News Scraper",
    goal="Scrape pharma newsroom websites for recent (last 14 days) breast cancer–related articles.",
    backstory="An Anthropic-powered agent that identifies current pharmaceutical updates on breast cancer.",
    tools=[custom_selenium_tool],
    verbose=True,
    llm=anthropic_llm
)
