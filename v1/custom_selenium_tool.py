# src/mslinsightsv1/agents/custom_selenium_tool.py
from crewai.tools import BaseTool
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import time, re

class CustomSeleniumScraper(BaseTool):
    name: str = "Custom Selenium Scraper"
    description: str = (
        "Scrapes pharma newsroom URLs using Selenium and returns breast cancer–related articles "
        "from the last 14 days with title, snippet, and publication date."
    )

    def _run(self, urls: list[str]):
        articles = []
        service = Service(r"C:\Chromedriver\chromedriver.exe")

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(service=service, options=options)
        cutoff = datetime.now() - timedelta(days=14)

        for url in urls:
            try:
                driver.get(url)
                time.sleep(3)
                html = driver.page_source
                soup = BeautifulSoup(html, "html.parser")

                title = driver.title or ""
                text = soup.get_text(" ", strip=True).lower()

                # ---- date extraction ----
                date_text = ""
                pub_date = None

                # <time> tag
                time_tag = soup.find("time")
                if time_tag and time_tag.get("datetime"):
                    try:
                        pub_date = datetime.fromisoformat(time_tag["datetime"][:10])
                    except Exception:
                        pass

                # fallback regex search
                if not pub_date:
                    match = re.search(r"(\d{1,2}\s+\w+\s+\d{4})", html)
                    if match:
                        try:
                            pub_date = datetime.strptime(match.group(1), "%d %B %Y")
                        except Exception:
                            pass

                if not pub_date or pub_date < cutoff:
                    continue  # skip old or undated pages

                # ---- keyword filter ----
                if "breast cancer" in text:
                    articles.append({
                        "source": url.split("/")[2],
                        "url": url,
                        "title": title.strip(),
                        "content": text[:1000],
                        "publication_date": pub_date.strftime("%Y-%m-%d")
                    })
                    print(f"✅ Found breast cancer article: {title}")

            except Exception as e:
                print(f"❌ Error scraping {url}: {e}")

        driver.quit()
        return articles
