# src/mslinsightsv1/tasks/scrape_task.py
from crewai import Task
from mslinsightsv1.agents.scraper_agent import scraper_agent

scraper_task = Task(
    description=(
        "Use the Custom Selenium Scraper to visit each pharma newsroom URL and extract "
        "articles mentioning 'breast cancer' published within the last 14 days. "
        "Return a JSON list with source, url, title, content, and publication_date."
    ),
    expected_output="A JSON list of recent breast cancer articles.",
    agent=scraper_agent,
)
