import os
import json
import re
from typing import List, Dict, Optional
from urllib.parse import urlparse

from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import tweepy

# Watsonx
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model

# =========================
# LOAD ENV
# =========================
load_dotenv()

# ---- Twitter creds ----
API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET = os.getenv("TWITTER_API_SECRET")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

if not all([API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET]):
    raise RuntimeError("Missing Twitter credentials in .env")

# ---- Watsonx creds ----
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_SPACE_ID = os.getenv("WATSONX_SPACE_ID", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")

if not (WATSONX_API_KEY and WATSONX_URL and (WATSONX_SPACE_ID or WATSONX_PROJECT_ID)):
    raise RuntimeError("Missing watsonx credentials in .env")

# =========================
# CONSTANTS / CONFIG
# =========================
LLAMA_MODEL = "meta-llama/llama-3-3-70b-instruct"
MAX_TWEET_CHARS = 280
MAX_POLL_OPTIONS = 4
MAX_POLLS_PER_ARTICLE = 4

# JSON input (articles with summary_280, url, entities, etc.)
INPUT_JSON = "article_summaries_extractions.json"

# Safety switch
DRY_RUN = False      # set to False to actually post
MAX_ARTICLES = 1    # how many articles to process/post

# =========================
# GLOBAL MODELS
# =========================
EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Twitter/X client (v2)
client = tweepy.Client(
    consumer_key=API_KEY,
    consumer_secret=API_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET,
)

# =========================
# TEXT HELPERS
# =========================
AVOID_PHRASES = [
    r"\bbreakthrough\b", r"\bmiracle\b", r"\bgame[- ]?changing\b",
    r"\bcure\b", r"\bguarantee(d)?\b", r"\bperfect\b", r"\bmust[- ]?have\b",
]

def normalize_ws(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s.strip())

def enforce_neutrality(s: str) -> str:
    for pat in AVOID_PHRASES:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    return s

def trim_to_limit(s: str, limit: int = MAX_TWEET_CHARS) -> str:
    if len(s) <= limit:
        return s
    return s[:limit - 1] + "…"

def parse_polls_block(text: str) -> List[str]:
    """
    Parse model output into blocks like:
    Q: ...
    - option1
    - option2
    """
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"\n(?=Q\d*\s*:|Question\s*:|Q\s*:)", text, flags=re.I)
    blocks = []
    for part in parts:
        p = part.strip()
        if p and re.match(r"^(Q\d*|Question|Q)\s*:", p, flags=re.I):
            blocks.append(p)
    if not blocks:
        return []
    cleaned = []
    for b in blocks:
        lines = [normalize_ws(ln) for ln in b.splitlines() if ln.strip()]
        if not lines:
            continue
        if not re.match(r"^Q\d*\s*:", lines[0], flags=re.I):
            lines[0] = "Q: " + re.sub(r"^(Q\d*|Question|Q)\s*:\s*", "", lines[0], flags=re.I)
        qline = lines[0]
        opts = [ln for ln in lines[1:] if ln.startswith("- ")]
        if len(opts) < 3:
            continue
        opts = opts[:MAX_POLL_OPTIONS]
        cleaned.append("\n".join([qline] + opts))
    return cleaned

def dedup_semantic(blocks: List[str], threshold: float) -> List[str]:
    if len(blocks) <= 1:
        return blocks
    embs = EMBEDDER.encode(blocks, normalize_embeddings=True)
    keep, kept = [], []
    for i, b in enumerate(blocks):
        if not kept:
            keep.append(b)
            kept.append(embs[i])
            continue
        sims = util.cos_sim(embs[i], np.stack(kept)).cpu().numpy()[0]
        if sims.max() < threshold:
            keep.append(b)
            kept.append(embs[i])
    return keep

# =========================
# JSON → ARTICLE SCHEMA
# =========================
def _coerce_list(x) -> List[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    return [str(x).strip()]

def adapt_article_schema(art: Dict) -> Dict:
    headline = (art.get("headline") or art.get("title") or "").strip()
    url = (art.get("url") or "").strip()
    content = (art.get("content") or art.get("article") or "").strip()
    summary = (art.get("summary_280") or art.get("summary_tweet") or art.get("summary") or "").strip()

    entities = art.get("entities") or {}
    drugs = _coerce_list(entities.get("drug_names") or art.get("drugs"))
    companies = _coerce_list(entities.get("company_name") or art.get("companies"))
    trial_names = _coerce_list(entities.get("trial_names"))
    trial_phases = _coerce_list(entities.get("trial_phases"))
    indications = _coerce_list(entities.get("indications"))

    grounding_text = content if content else (headline + "\n\n" + summary)

    return {
        "headline": headline,
        "url": url,
        "content": content,
        "summary": summary,
        "entities": {
            "drug_names": drugs,
            "company_name": companies,
            "trial_names": trial_names,
            "trial_phases": trial_phases,
            "indications": indications,
        },
        "grounding_text": grounding_text,
    }

# =========================
# SIMPLE AWARENESS POLL
# =========================
def build_awareness_poll(json_entities: Dict) -> str:
    drugs = json_entities.get("drug_names") or []
    inds = json_entities.get("indications") or []
    drug = drugs[0] if drugs else None
    ind = inds[0] if inds else None

    if drug and ind:
        q = f"Q: Were you aware of this {drug} development for {ind} before reading?"
    elif ind:
        q = f"Q: Were you aware of this {ind} development before reading?"
    else:
        q = "Q: Were you aware of this development before reading?"

    poll = q + "\n- Yes, was aware\n- No, new information\n- Somewhat aware\n- Will research further"
    poll = enforce_neutrality(poll)
    poll = trim_to_limit(poll, MAX_TWEET_CHARS)
    return poll

# =========================
# POLL MINER (watsonx Llama)
# =========================
class PollMiner:
    def __init__(self, model_name: str = LLAMA_MODEL):
        self.model_name = model_name

        creds = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
        params = {
            "decoding_method": "sample",
            "temperature": 0.8,
            "top_p": 0.9,
            "max_new_tokens": 480,
        }
        kwargs = {}
        token = (WATSONX_SPACE_ID or WATSONX_PROJECT_ID).strip()
        if WATSONX_SPACE_ID:
            kwargs["space_id"] = token
        else:
            kwargs["project_id"] = token

        self.wxa_model = Model(
            model_id=model_name,
            credentials=creds,
            params=params,
            **kwargs,
        )

    def _format_prompt(self, headline: str, url: str, grounding_text: str, key_facts: Dict) -> str:
        snippet = grounding_text.strip()
        if len(snippet) > 1600:
            snippet = snippet[:1600]

        facts_payload = {
            "headline": headline,
            "url": url,
            "entities": {
                "drug_names": key_facts.get("drug_names", [])[:6],
                "company_name": key_facts.get("company_name", [])[:6],
                "trial_names": key_facts.get("trial_names", [])[:6],
                "trial_phases": key_facts.get("trial_phases", [])[:6],
                "indications": key_facts.get("indications", [])[:6],
            }
        }
        facts_str = json.dumps(facts_payload, ensure_ascii=False)

        return f"""
You are an assistant that writes Twitter/X poll questions to understand a **doctor's (HCP) perspective** on ONE breast cancer news article.

STRICT CONTENT RULES — READ CAREFULLY:
- USE ONLY information that appears in the headline or article content below (or in Key Facts, which directly mirrors the provided JSON). Do NOT invent drugs, companies, endpoints, biomarkers, or settings.
- Each poll MUST clearly reference the specific context present in the article (e.g., drug name, indication, phase/setting, relevant endpoint if mentioned).
- Produce AT MOST **4** polls and make them mutually DISTINCT in focus (avoid paraphrases):
  • practice impact (practice-informing vs practice-changing)
  • intent to use in practice
  • patient selection / setting (line of therapy, subtype, biomarker)
  • endpoints / treatment discussions (e.g., PFS/OS/ORR if mentioned)
- Keep each poll UNDER 280 characters total (question + options) and include 3–4 concise options.
- Return COMPLETE polls only (no truncated options).
- Purpose: elicit the **doctor's perspective** (practice intent), not generic Q&A.

KEY FACTS (from JSON; do not add new facts):
{facts_str}

ARTICLE CONTENT (truncated if long):
{snippet}

OUTPUT FORMAT (up to 4 polls; no extra text):
Q: <HCP-perspective question that strictly reflects article content>
- Option 1
- Option 2
- Option 3
- Option 4 (optional)
"""

    def _gen_watsonx(self, prompt: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
        try:
            self.wxa_model.params.update({
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_new_tokens": int(max_new_tokens),
                "decoding_method": "sample",
            })
        except Exception:
            pass

        result = self.wxa_model.generate_text(prompt=prompt)
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            if "results" in result and result["results"]:
                cand = result["results"][0].get("generated_text") or result["results"][0].get("text")
                if cand:
                    return cand
            return result.get("generated_text") or json.dumps(result)
        return str(result)

    def generate(self, article: Dict,
                 passes: int = 3,
                 temperature: float = 0.8,
                 top_p: float = 0.9,
                 max_new_tokens: int = 480) -> List[str]:

        all_blocks: List[str] = []

        for _ in range(passes):
            prompt = self._format_prompt(
                article["headline"],
                article["url"],
                article["grounding_text"],
                article["entities"],
            )
            out = self._gen_watsonx(prompt, temperature, top_p, max_new_tokens)
            blocks = parse_polls_block(out)
            for b in blocks:
                b = enforce_neutrality(b)
                b = trim_to_limit(b, MAX_TWEET_CHARS)
                all_blocks.append(b)

        # basic filtering
        valid = []
        for blk in all_blocks:
            lines = [ln for ln in blk.splitlines() if ln.strip()]
            if not lines or not lines[0].lower().startswith("q"):
                continue
            opts = [ln for ln in lines[1:] if ln.startswith("- ")]
            if len(opts) < 3:
                continue
            if len("\n".join(lines)) > MAX_TWEET_CHARS:
                continue
            valid.append(blk)

        valid = dedup_semantic(valid, threshold=0.93)
        return valid[:MAX_POLLS_PER_ARTICLE]

# =========================
# TWITTER POSTING HELPERS
# =========================
def safe_trim(text: str) -> str:
    return text if len(text) <= 280 else text[:277] + "..."

def build_tweet1(item: dict) -> str:
    url = (item.get("url") or "").strip()
    entities = item.get("entities") or {}
    company = ""
    # company_name may be list in your JSON; handle both
    cname = entities.get("company_name")
    if isinstance(cname, list):
        company = ", ".join([c for c in cname if c])
    elif isinstance(cname, str):
        company = cname.strip()

    parts = []
    if url:
        parts.append(url)
    if company:
        parts.append(company)
    return "\n".join(parts)

def build_tweet2(item: dict) -> str:
    summary = (item.get("summary") or item.get("summary_280") or "").strip()
    return safe_trim(summary)

def parse_poll_block_for_twitter(block: str) -> tuple[str, List[str]]:
    """
    Convert block:
      Q: ....
      - opt1
      - opt2
    into (question, [opt1, opt2, ...])
    """
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return "", []
    qline = lines[0]
    q = re.sub(r"^Q\s*:\s*", "", qline, flags=re.I).strip()
    options = []
    for ln in lines[1:]:
        if ln.startswith("- "):
            options.append(ln[2:].strip())
    return q, options

def post_thread_for_article(article: Dict, polls: List[str], dry_run: bool = True):
    """
    Thread:
      Tweet 1: URL + Company
      Tweet 2: Summary
      Tweet 3+: polls as Twitter polls
    """
    tweet1 = build_tweet1(article)
    tweet2 = build_tweet2(article)

    print("\n=== NEW THREAD ===")
    print("Tweet 1:\n", tweet1)
    print("Tweet 2:\n", tweet2)
    print("Num polls:", len(polls))

    if dry_run:
        for i, p in enumerate(polls, start=1):
            print(f"--- Poll {i} ---\n{p}\n")
        print("DRY_RUN=True → No tweets posted.")
        return

    # 1) Post parent tweet
    resp1 = client.create_tweet(text=tweet1)
    parent_id = resp1.data["id"]
    print("Posted Tweet 1, id:", parent_id)

    # 2) Post summary as reply
    resp2 = client.create_tweet(
        text=tweet2,
        in_reply_to_tweet_id=parent_id
    )
    current_parent = resp2.data["id"]
    print("Posted Tweet 2, id:", current_parent)

    # 3) Post each poll as reply to previous tweet
    for idx, block in enumerate(polls, start=1):
        question, options = parse_poll_block_for_twitter(block)
        if not question or len(options) < 2:
            print(f"Skipping poll {idx}: invalid question or options.")
            continue

        print(f"Posting poll {idx}: {question} | options={options}")
        resp_poll = client.create_tweet(
            text=question,
            in_reply_to_tweet_id=current_parent,
            poll_options=options[:MAX_POLL_OPTIONS],
            poll_duration_minutes=1440,   # 1 day
        )
        current_parent = resp_poll.data["id"]
        print("  → Poll tweet id:", current_parent)

# =========================
# MAIN PIPELINE
# =========================
def main():
    # load articles
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("Input JSON must be a list of article dicts.")

    miner = PollMiner()

    for idx, art in enumerate(raw[:MAX_ARTICLES], start=1):
        print(f"\n=== ARTICLE {idx} ===")
        adapted = adapt_article_schema(art)

        awareness = build_awareness_poll(adapted["entities"])
        llm_polls = miner.generate(adapted)

        polls = [awareness] + llm_polls
        # dedup again on full set
        polls = list(dict.fromkeys([normalize_ws(p) for p in polls]))
        polls = dedup_semantic(polls, threshold=0.93)
        polls = polls[:MAX_POLLS_PER_ARTICLE]

        post_thread_for_article(
            article=adapted,
            polls=polls,
            dry_run=DRY_RUN
        )

if __name__ == "__main__":
    main()
