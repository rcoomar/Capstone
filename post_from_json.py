import os
import json
import tweepy
from dotenv import load_dotenv

# ==========================================
# LOAD SECRETS FROM .env
# ==========================================
load_dotenv()

API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET = os.getenv("TWITTER_API_SECRET")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

if not all([API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET]):
    raise RuntimeError("Missing Twitter credentials. Check your .env file.")

# ==========================================
# CONFIG
# ==========================================
JSON_PATH = "article_summaries_extractions.json"
MAX_ITEMS = 1
DRY_RUN = False  # set False to actually post


# ==========================================
# TWITTER AUTH – v2 Client
# ==========================================
client = tweepy.Client(
    consumer_key=API_KEY,
    consumer_secret=API_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET,
)


# ==========================================
# HELPERS
# ==========================================
def safe_trim(text: str) -> str:
    """Hard cap at 280 characters just in case."""
    return text if len(text) <= 280 else text[:277] + "..."


def build_tweet_pair(item: dict) -> tuple[str, str]:
    """
    Build TWO tweets:

    Tweet 1: URL + Company
    Tweet 2: FULL summary

    Only summary may be trimmed to 280 chars if extremely long.
    """

    url = (item.get("url") or "").strip()
    summary = (item.get("summary_280") or "").strip()

    entities = item.get("entities") or {}
    company = (entities.get("company_name") or "").strip()

    # ---- Tweet 1 (URL + company) ----
    parts1 = []
    if url:
        parts1.append(url)
    if company:
        parts1.append(company)

    tweet1 = "\n".join(parts1)
    tweet1 = safe_trim(tweet1)  # unlikely to exceed 280, but safe

    # ---- Tweet 2 (Summary) ----
    tweet2 = safe_trim(summary)  # should be <=280 normally

    return tweet1, tweet2


# ==========================================
# MAIN
# ==========================================
def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    for i, item in enumerate(items):
        if i >= MAX_ITEMS:
            break

        tweet1, tweet2 = build_tweet_pair(item)

        print(f"\n=== ITEM #{i+1} ===")
        print("--- Tweet 1 ---")
        print(tweet1)
        print("--- Tweet 2 ---")
        print(tweet2)
        print("=" * 40)

        if not DRY_RUN:
            # 1) Post parent tweet
            resp1 = client.create_tweet(text=tweet1)
            parent_id = resp1.data["id"]

            # 2) Post summary as a reply (thread)
            resp2 = client.create_tweet(
                text=tweet2,
                in_reply_to_tweet_id=parent_id
            )

            print(f"Posted thread ✔️ (tweet_id={parent_id})")

    if DRY_RUN:
        print("\nDRY_RUN = True — No tweets posted.")


if __name__ == "__main__":
    main()
