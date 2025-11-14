# src/mslinsightsv1/main.py
from crewai import Crew
import pandas as pd, json
from mslinsightsv1.agents.scraper_agent import scraper_agent
from mslinsightsv1.tasks.scrape_task import scraper_task

# Load URLs
df = pd.read_csv(r"C:\Sem4V1\mslinsightsv1\data\pharma_urls.csv")
urls = df["url"].dropna().tolist()

# Create Crew
crew = Crew(
    agents=[scraper_agent],
    tasks=[scraper_task],
    verbose=True
)

inputs = {"urls": urls}
result = crew.kickoff(inputs=inputs)

# Save
output_path = r"C:\Sem4V1\mslinsightsv1\data\breast_cancer_articles.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"✅ Scraping completed and saved to {output_path}")
