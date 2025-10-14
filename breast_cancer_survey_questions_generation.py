# app.py
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlparse

import numpy as np
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import login  # Added import

# =========================
# HUGGING FACE AUTHENTICATION
# =========================
# Add this section at the top of your sidebar or in a secure configuration section
with st.sidebar:
    st.title("🔐 Hugging Face Setup")
    hf_token = st.text_input(
        "Hugging Face API Token",
        type="password",
        help="Enter your Hugging Face token to use BioMistral model"
    )
    if hf_token:
        try:
            login(token=hf_token)
            st.success("✅ Successfully authenticated with Hugging Face!")
        except Exception as e:
            st.error(f"❌ Authentication failed: {e}")

# =========================
# DEFAULTS - UPDATED MODEL
# =========================
DEFAULT_MODEL_NAME = "mistralai/BioMistral-7B"  # Updated model name
MAX_NEW_TOKENS = 200
MAX_TWEET_CHARS = 280
MAX_POLL_OPTIONS = 4

# =========================
# UI SIDEBAR - UPDATED MODEL OPTIONS
# =========================
st.set_page_config(page_title="Breast Cancer News → Poll Generator", layout="wide")

with st.sidebar:
    st.title("⚙️ Settings")
    model_name = st.selectbox(
        "LLM Model:",
        [
            "mistralai/BioMistral-7B",  # Primary choice
            "google/flan-t5-base",      # Fallback option
            "google/flan-t5-large"      # Fallback option
        ],
        index=0,
        help="BioMistral for medical content, Flan-T5 as fallback. BioMistral requires HF token."
    )
    passes_per_article = st.slider("Generation passes per article", 1, 6, 3)
    temperature = st.slider("Temperature (creativity)", 0.0, 1.2, 0.8, 0.1)
    top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.9, 0.05)
    similarity_dup_threshold = st.slider("Semantic de-dup threshold", 0.80, 0.99, 0.90, 0.01,
                                         help="Higher = keep only more distinct polls")
    st.markdown("---")
    st.caption("Upload a JSON list like: `[{'headline','url','content'}, ...]`")

# =========================
# CACHING HEAVY OBJECTS - UPDATED FOR BIO MISTRAL
# =========================
@st.cache_resource(show_spinner=False)
def get_text_generation_pipeline(model_name: str):
    """Get appropriate pipeline based on model type"""
    if "BioMistral" in model_name or "mistral" in model_name.lower():
        # Use text-generation pipeline for causal LM
        return pipeline(
            "text-generation",
            model=model_name,
            tokenizer=AutoTokenizer.from_pretrained(model_name),
            device_map="auto",
            torch_dtype="auto"  # Automatically handle dtype
        )
    else:
        # Fallback to text2text for Flan models
        return pipeline("text2text-generation", model=model_name, device_map="auto")

@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# =========================
# TEXT HELPERS
# =========================
AVOID_PHRASES = [
    r"\bbreakthrough\b", r"\bmiracle\b", r"\bgame[- ]?changing\b",
    r"\bcure\b", r"\bguarantee(d)?\b", r"\bperfect\b", r"\bmust[- ]?have\b"
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
    lines = s.splitlines()
    if not lines:
        return s[:limit]
    q = lines[0]
    opts = [ln for ln in lines[1:] if ln.strip()]
    while len("\n".join([q] + opts)) > limit and len(q) > 40:
        q = q[:-10].rstrip(" .,:;")
    def shorten(op):
        return (op[:80] + "…") if len(op) > 80 else op
    opts = [shorten(op) for op in opts]
    if len("\n".join([q] + opts)) > limit and len(opts) > 3:
        opts = opts[:3]
    out = "\n".join([q] + opts)
    return out[:limit]

def parse_polls_block(text: str) -> List[str]:
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
        lines = [ln.rstrip() for ln in text.splitlines()]
        cur = []
        collecting = False
        for ln in lines:
            if ln.strip().endswith("?"):
                if cur:
                    blocks.append("\n".join(cur)); cur = []
                cur = [f"Q: {normalize_ws(ln)}"]; collecting = True
            elif collecting and re.match(r"^\s*([-•—]|\d+\.|[A-D]\))", ln):
                cur.append("- " + re.sub(r"^\s*([-•—]|\d+\.|[A-D]\))\s*", "", ln).strip())
            elif collecting and ln.strip() == "":
                if cur:
                    blocks.append("\n".join(cur)); cur = []
                collecting = False
        if cur:
            blocks.append("\n".join(cur))
    cleaned = []
    for b in blocks:
        lines = [normalize_ws(ln) for ln in b.splitlines() if ln.strip()]
        if not lines:
            continue
        if not re.match(r"^Q\d*\s*:", lines[0], flags=re.I):
            lines[0] = "Q: " + re.sub(r"^(Q\d*|Question|Q)\s*:\s*", "", lines[0], flags=re.I)
        qline = lines[0]
        opts = [ln for ln in lines[1:] if re.match(r"^(-|•|—|\d+\.|[A-D]\))\s*", ln)]
        opts = ["- " + re.sub(r"^(-|•|—|\d+\.|[A-D]\))\s*", "", o).strip() for o in opts]
        if len(opts) < 3:
            continue
        opts = opts[:MAX_POLL_OPTIONS]
        cleaned.append("\n".join([qline] + opts))
    return cleaned

def dedup_semantic(blocks: List[str], threshold: float, embedder) -> List[str]:
    if len(blocks) <= 1:
        return blocks
    embs = embedder.encode(blocks, normalize_embeddings=True)
    keep, kept = [], []
    for i, b in enumerate(blocks):
        if not kept:
            keep.append(b); kept.append(embs[i]); continue
        sims = util.cos_sim(embs[i], np.stack(kept)).cpu().numpy()[0]
        if sims.max() < threshold:
            keep.append(b); kept.append(embs[i])
    return keep

# =========================
# ENTITY/FACT EXTRACTION
# =========================
KNOWN_PHARMA = {
    "Genentech","Roche","Novartis","Pfizer","AstraZeneca","Eli Lilly","Merck",
    "Sanofi","Gilead","BMS","Amgen","GSK","Bayer","Takeda","Boehringer Ingelheim",
    "BeiGene","Seagen","Roche Genentech","Sermonix Pharma","Sermonix"
}
ORG_SUFFIXES = r"(?: Inc\.?| Corp\.?| Corporation| Ltd\.?| LLC| plc| AG| SA| NV| Co\.?)"

def company_from_url(url: str) -> Optional[str]:
    try:
        host = urlparse(url).netloc.lower()
        if not host:
            return None
        for prefix in ("www.", "amp.", "m.", "news."):
            if host.startswith(prefix):
                host = host[len(prefix):]
        mapping = {
            "sermonixpharma.com": "Sermonix Pharma",
            "gene.com": "Genentech",
            "roche.com": "Roche",
            "novartis.com": "Novartis",
            "pfizer.com": "Pfizer",
            "astrazeneca.com": "AstraZeneca",
            "lilly.com": "Eli Lilly",
            "merck.com": "Merck",
            "sanofi.com": "Sanofi",
            "gilead.com": "Gilead",
            "bms.com": "BMS",
            "amgen.com": "Amgen",
            "gsk.com": "GSK",
            "bayer.com": "Bayer",
            "takeda.com": "Takeda",
            "boehringer-ingelheim.com": "Boehringer Ingelheim",
            "beigene.com": "BeiGene",
            "seagen.com": "Seagen",
        }
        if host in mapping:
            return mapping[host]
        parts = host.split(".")
        brand = parts[-2] if len(parts) >= 2 else parts[0]
        brand = brand.replace("-", " ").strip()
        if not brand:
            return None
        return brand[0].upper() + brand[1:]
    except Exception:
        return None

def find_companies(text: str) -> List[str]:
    found = set()
    for k in KNOWN_PHARMA:
        if re.search(rf"\b{k}\b", text, flags=re.I):
            found.add(k)
    for m in re.finditer(rf"\b([A-Z][A-Za-z&\-]+(?:\s+[A-Z][A-Za-z&\-]+){{0,2}})(?:{ORG_SUFFIXES})?\b", text):
        name = m.group(0).strip()
        if len(name.split()) >= 1 and not re.match(r"^(Q|FDA|HR\+|HER2|\(?HR\+|\(?HER2|PIK3CA|Phase|Breast|Cancer|Trial|Study)$", name):
            found.add(name)
    out = [re.sub(r"\s+", " ", f).strip() for f in found]
    return list({x.lower(): x for x in out}.values())

def pick_drug(text: str, lower_text: str) -> Optional[str]:
    m = re.search(r"\b([A-Z][A-Za-z0-9\-]{3,}(?:™|®))", text)
    if m: return m.group(1)
    m = re.search(r"\b([A-Z][A-Za-z0-9\-]{3,})\s*\(([^)]+)\)", text)
    if m: return f"{m.group(1)} ({m.group(2)})"
    m = re.search(r"\b(inavolisib|palbociclib|fulvestrant|trastuzumab|pertuzumab|t-?dm1|olaparib|abemaciclib|ribociclib|alpelisib|everolimus|letrozole|anastrozole|exemestane)\b", lower_text)
    if m: return m.group(1)
    return None

def pick_indication(text: str) -> str:
    m = re.search(r"(HR\+\s*[,/]*\s*HER2-\s*.*?breast cancer|HER2\+\s*.*?breast cancer|HER2-\s*.*?breast cancer|HR\+\s*.*?breast cancer|metastatic breast cancer|early breast cancer|breast cancer)", text, flags=re.I)
    return m.group(1) if m else "breast cancer"

def pick_phase(lower_text: str) -> Optional[str]:
    m = re.search(r"\bphase\s*(I{1,3}|IV|V|\d)\b", lower_text)
    return m.group(1).upper() if m else None

def detect_topics(text: str, lower_text: str) -> Dict[str, bool]:
    return dict(
        fda_approval=("fda" in lower_text and re.search(r"\bapprov|\bauthoriz", lower_text)),
        trial=bool(re.search(r"\b(clinical )?trial\b|\bstudy\b", lower_text)),
        acquisition=bool(re.search(r"\bacquisit|acquire|merger\b", lower_text)),
        partnership=bool(re.search(r"\b(partner|collaborat|licens|alliance)\w*\b", lower_text)),
        endpoints=bool(re.search(r"\bprogression[- ]free survival\b|\bPFS\b|\boverall survival\b|\bOS\b|\bORR\b|\bresponse rate\b", text, flags=re.I)),
        setting_firstline=bool(re.search(r"\bfirst[- ]?line\b", lower_text)),
        setting_adjuvant=bool(re.search(r"\badjuvant\b", lower_text)),
        mutation_pik3ca=("pik3ca" in lower_text),
        receptor_hrpos=("hr+" in lower_text or "hormone receptor-positive" in lower_text),
        her2_neg=("her2-" in lower_text or "her2-negative" in lower_text),
        her2_pos=("her2+" in lower_text or "her2-positive" in lower_text),
    )

def extract_facts(headline: str, content: str, url: str) -> Dict:
    text = f"{headline or ''}\n{content or ''}"
    text_norm = re.sub(r"\s+", " ", text).strip()
    t = text_norm.lower()
    companies = find_companies(text_norm)
    pub = company_from_url(url) or None
    if pub and pub not in companies:
        companies.insert(0, pub)
    drug = pick_drug(text_norm, t)
    indication = pick_indication(text_norm)
    phase = pick_phase(t)
    topics = detect_topics(text_norm, t)
    return dict(
        companies=companies,
        publisher=pub,
        drug=drug,
        indication=indication,
        phase=phase,
        topics=topics
    )

# =========================
# AWARENESS THEME & POLL
# =========================
def detect_theme_keyphrase(headline: str, content: str, url: str) -> str:
    facts = extract_facts(headline, content, url)
    drug, ind, phase = facts["drug"], facts["indication"], facts["phase"]
    companies, pub = facts["companies"], facts["publisher"]
    tp = facts["topics"]

    if tp["fda_approval"]:
        if drug: return f"FDA approval of {drug} for {ind}"
        return f"FDA approval for {ind}"

    if tp["trial"]:
        if phase and drug and companies:
            return f"phase {phase} clinical trial of {drug} by {companies[0]}"
        if phase and drug:
            return f"phase {phase} clinical trial of {drug}"
        if phase and companies:
            return f"phase {phase} clinical trial by {companies[0]}"
        if phase:
            return f"phase {phase} clinical trial"
        return "clinical trial"

    if tp["acquisition"]:
        acq = re.search(r"([A-Z][A-Za-z&\-\s]{2,50})\s+acquire[sd]?\s+([A-Z][A-Za-z0-9&\-\s]{2,50})", f"{headline}\n{content}")
        if acq:
            acquirer = acq.group(1).strip(); target = acq.group(2).strip()
            return f"acquisition of {target} by {acquirer}"
        uniq = companies.copy()
        if pub and pub not in uniq: uniq.append(pub)
        if len(uniq) >= 2:
            return f"acquisition involving {uniq[0]} and {uniq[1]}"

    if tp["partnership"]:
        uniq = companies.copy()
        if pub and pub not in uniq: uniq.append(pub)
        if len(uniq) >= 2:
            return f"partnership between {uniq[0]} and {uniq[1]}"
        return "industry partnership"

    if tp["endpoints"]:
        if phase:
            return f"{'phase ' + phase + ' ' if phase else ''}trial results in {ind}"
        return f"clinical study results in {ind}"

    headline_short = re.sub(r"\s+", " ", (headline or "")).strip()
    if not headline_short or len(headline_short) > 60:
        headline_short = "breast cancer"
    if pub:
        return f"{headline_short} overview by {pub}"
    return headline_short

def build_awareness_poll(headline: str, content: str, url: str) -> str:
    """Updated awareness poll to be more practice-focused"""
    facts = extract_facts(headline, content, url)
    drug, ind = facts["drug"], facts["indication"]
    
    if drug and ind:
        q = f"Q: Were you aware of this {drug} development for {ind} before reading?"
    elif ind:
        keyphrase = detect_theme_keyphrase(headline, content, url)
        q = f"Q: Were you aware of this {ind} development before reading?"
    else:
        keyphrase = detect_theme_keyphrase(headline, content, url)
        q = f"Q: Were you aware of this development before reading?"
    
    poll = q + "\n- Yes, was aware\n- No, new information\n- Somewhat aware\n- Will research further"
    poll = enforce_neutrality(poll)
    poll = trim_to_limit(poll, MAX_TWEET_CHARS)
    return poll

# =========================
# LLM MINER - UPDATED FOR BIO MISTRAL WITH PRACTICE FOCUS
# =========================
class PollMiner:
    def __init__(self, model_name: str):
        self.gen_pipeline = get_text_generation_pipeline(model_name)
        self.is_bio_mistral = "BioMistral" in model_name or "mistral" in model_name.lower()

    def _format_bio_mistral_prompt(self, headline: str, url: str, content: str, facts: Dict) -> str:
        """Format prompt specifically for BioMistral with practice-focused questions"""
        snippet = content.strip()
        if len(snippet) > 1400:
            snippet = snippet[:1400]
        
        facts_str = json.dumps({
            "drug": facts["drug"],
            "companies": facts["companies"][:3],
            "indication": facts["indication"],
            "phase": facts["phase"],
            "topics_true": [k for k,v in facts["topics"].items() if v]
        }, ensure_ascii=False)
        
        prompt = f"""<s>[INST] You are a medical content specialist creating Twitter/X poll questions about breast cancer news.

TASK: Generate practice-focused Twitter/X poll questions that assess how this medical information impacts clinical practice.

CRITICAL RULES:
- Focus on PRACTICE IMPACT: information utility, practice changes, clinical application
- Use these question frameworks:
  * "Is this information practice-informing for [specific context]?"
  * "Is this information practice-changing for [specific context]?" 
  * "Will you use this information in your practice for [specific context]?"
  * "How does this impact your clinical approach to [specific context]?"
- Every question MUST explicitly mention specific clinical context from the facts (drug, indication, patient population, setting)
- Use ONLY information from the provided facts and article content.
- No invented details, no promotional language, no medical advice.
- Each poll must be under 280 characters including the question and options.
- Provide 3-4 concise, clinically relevant options.

ARTICLE INFORMATION:
Headline: "{headline}"
URL: {url}
Key Facts: {facts_str}

ARTICLE CONTENT:
{snippet}

Generate several DISTINCT practice-focused poll questions. Format each poll as:
Q: <practice-focused question with specific clinical context>
- Option 1
- Option 2  
- Option 3
- Option 4 (optional)

Ensure each question assesses clinical utility and practice impact. [/INST]"""
        return prompt

    def _format_flan_prompt(self, headline: str, url: str, content: str, facts: Dict) -> str:
        """Updated prompt format for Flan models with practice focus"""
        snippet = content.strip()
        if len(snippet) > 1400:
            snippet = snippet[:1400]
        facts_str = json.dumps({
            "drug": facts["drug"],
            "companies": facts["companies"][:3],
            "indication": facts["indication"],
            "phase": facts["phase"],
            "topics_true": [k for k,v in facts["topics"].items() if v]
        }, ensure_ascii=False)
        
        return f"""Generate practice-focused Twitter/X poll questions assessing clinical utility and practice impact.

FOCUS AREAS:
- Is this information practice-informing?
- Is this information practice-changing?
- Will you use this in your practice?
- How does this impact clinical approach?

RULES:
- Every question must focus on PRACTICE IMPACT and mention specific clinical context from facts
- Use specific clinical details: {facts_str}
- No generic questions - always tie to specific drugs, indications, or clinical scenarios
- ≤ 280 chars including options. Provide 3–4 clinically relevant options.

Headline: "{headline}"
URL: {url}
Article:
\"\"\"{snippet}\"\"\"

Output practice-focused poll blocks. Format:
Q: <practice-impact question with specific context>
- Option
- Option
- Option
- Option (optional)
"""

    def _prompt(self, headline: str, url: str, content: str, facts: Dict) -> str:
        if self.is_bio_mistral:
            return self._format_bio_mistral_prompt(headline, url, content, facts)
        else:
            return self._format_flan_prompt(headline, url, content, facts)

    def generate(self, headline: str, url: str, content: str, facts: Dict,
                 passes: int, temperature: float, top_p: float) -> List[str]:
        all_blocks: List[str] = []
        
        for _ in range(passes):
            try:
                if self.is_bio_mistral:
                    # BioMistral specific generation
                    result = self.gen_pipeline(
                        self._prompt(headline, url, content, facts),
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        num_return_sequences=1,
                        pad_token_id=self.gen_pipeline.tokenizer.eos_token_id,
                        return_full_text=False  # Don't repeat the prompt
                    )
                    out = result[0]["generated_text"]
                else:
                    # Original Flan-T5 generation
                    result = self.gen_pipeline(
                        self._prompt(headline, url, content, facts),
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        num_return_sequences=1,
                    )
                    out = result[0]["generated_text"]
                
                blocks = parse_polls_block(out)
                for b in blocks:
                    b = enforce_neutrality(b)
                    b = trim_to_limit(b, MAX_TWEET_CHARS)
                    all_blocks.append(b)
                    
            except Exception as e:
                st.warning(f"Generation attempt failed: {e}")
                continue

        # Exact dedup
        uniq = list(dict.fromkeys([normalize_ws(b) for b in all_blocks]))
        
        # Semantic dedup
        embedder = get_embedder()
        uniq = dedup_semantic(uniq, similarity_dup_threshold, embedder)

        # Enhanced grounded filter with practice focus
        grounded_terms = set()
        if facts["drug"]: grounded_terms.add(str(facts["drug"]).lower())
        for c in facts["companies"]:
            grounded_terms.add(c.lower())
        for token in [
            facts["indication"], facts["phase"],
            "pik3ca", "hr+", "her2-", "her2+", "pfs", "overall survival", "os", "adjuvant", "first-line"
        ]:
            if token: grounded_terms.add(str(token).lower())

        def is_grounded(poll: str) -> bool:
            t = poll.lower()
            # Check for specific clinical terms AND practice-focused language
            clinical_terms_present = any(term in t for term in grounded_terms)
            practice_focus_present = any(phrase in t for phrase in [
                "practice", "clinical", "use this", "impact", "change", "inform", 
                "applicable", "relevant", "approach", "discussion", "treatment"
            ])
            return clinical_terms_present and practice_focus_present

        final = []
        for blk in uniq:
            lines = [ln for ln in blk.splitlines() if ln.strip()]
            if not lines or not lines[0].lower().startswith("q"):
                continue
            opts = [ln for ln in lines[1:] if ln.startswith("- ")]
            if len(opts) < 3:
                continue
            if not is_grounded(blk):
                continue
            if len("\n".join(lines)) <= MAX_TWEET_CHARS:
                final.append(blk)
        return final

# =========================
# RULE-BASED FALLBACK - UPDATED FOR PRACTICE FOCUS
# =========================
def rule_based_polls(facts: Dict) -> List[str]:
    """Generate practice-focused rule-based polls"""
    polls: List[str] = []
    def add(q, options):
        block = "Q: " + q.strip()
        opts = ["- " + o.strip() for o in options if o.strip()]
        if len(opts) >= 3:
            block = enforce_neutrality(block + "\n" + "\n".join(opts))
            block = trim_to_limit(block, MAX_TWEET_CHARS)
            polls.append(block)

    drug, ind, phase = facts["drug"], facts["indication"], facts["phase"]
    companies = facts["companies"]
    tp = facts["topics"]

    # Practice-focused questions with specific clinical context
    if drug and ind:
        add(
            f"Is this information about {drug} for {ind} practice-changing for you?",
            ["Yes, will change my practice", "Informative but no practice change", "Not relevant to my practice", "Need more data"]
        )
        add(
            f"Will you use this {drug} information in your {ind} practice?",
            ["Yes, immediately applicable", "Yes, with certain patients", "No, not applicable", "Need guideline updates first"]
        )
    
    if tp["fda_approval"] and drug:
        add(
            f"How practice-informing is the FDA approval of {drug} for {ind}?",
            ["Highly practice-changing", "Moderately informative", "Minimal impact", "Awaiting real-world data"]
        )
    
    if tp["trial"] and phase and drug:
        add(
            f"Does this Phase {phase} {drug} trial data impact your clinical approach to {ind}?",
            ["Yes, changes my thinking", "Yes, confirms current approach", "No, not convincing", "Need more follow-up"]
        )
    
    if tp["endpoints"] and drug:
        add(
            f"How will these {drug} {ind} results influence your treatment discussions?",
            ["Significantly change discussions", "Modestly inform discussions", "No impact on discussions", "Uncertain until published"]
        )
    
    if companies and drug:
        add(
            f"Is this {drug} development by {companies[0]} practice-relevant for your {ind} patients?",
            ["Yes, highly relevant", "Yes, for some patients", "No, not applicable", "Depends on access/cost"]
        )

    # General practice-focused questions as fallback
    if ind and not polls:
        add(
            f"How practice-changing is this {ind} information?",
            ["Practice-changing", "Practice-informing", "Not practice-impacting", "Uncertain impact"]
        )

    embedder = get_embedder()
    polls = dedup_semantic(list(dict.fromkeys(polls)), 0.92, embedder)
    return polls

# =========================
# PIPELINE (one article)
# =========================
def build_polls_for_article(headline: str, url: str, content: str) -> List[str]:
    facts = extract_facts(headline, content, url)
    awareness = build_awareness_poll(headline, content, url)
    miner = PollMiner(model_name)
    llm_polls = miner.generate(headline, url, content, facts,
                               passes_per_article, temperature, top_p)
    if not llm_polls:
        llm_polls = rule_based_polls(facts)
    all_polls = [awareness] + llm_polls
    all_polls = list(dict.fromkeys([normalize_ws(p) for p in all_polls]))
    embedder = get_embedder()
    all_polls = dedup_semantic(all_polls, 0.92, embedder)
    if awareness in all_polls and all_polls[0] != awareness:
        all_polls.remove(awareness)
        all_polls.insert(0, awareness)
    return all_polls, facts

# =========================
# FILE INPUTS (repo root only)
# =========================
st.header("📰 Breast Cancer News → 🗳️ Poll Generator")

# 1) Upload
uploaded = st.file_uploader("Upload JSON (list of objects with 'headline','url','content')", type=["json"])

# 2) Discover JSON files in the repo root only
APP_DIR = Path(__file__).parent.resolve()
repo_files: List[Path] = sorted([p for p in APP_DIR.glob("*.json") if p.is_file()])
repo_choices = ["(none)"] + [p.name for p in repo_files]
repo_pick = st.selectbox("…or pick a JSON file from this repo (root)", options=repo_choices, index=0)

# 3) Manual path (optional)
path_input = st.text_input("…or enter a JSON file path (relative to repo root or absolute)", value="")

run_btn = st.button("Generate Polls", type="primary")

def load_json_any(source_file: Path) -> List[Dict]:
    data = json.loads(source_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of article dicts.")
    return data

articles: List[Dict] = []
if run_btn:
    try:
        if uploaded is not None:
            articles = json.load(uploaded)
        elif repo_pick != "(none)":
            chosen = APP_DIR / repo_pick
            articles = load_json_any(chosen)
        elif path_input.strip():
            p = Path(path_input)
            if not p.is_absolute():
                p = APP_DIR / p  # treat as relative to repo root
            if not p.exists():
                st.error(f"Path not found: {p}")
            else:
                articles = load_json_any(p)
        else:
            # Try a couple of common names in repo root
            guesses = [
                APP_DIR / "twitter_polls_generated_v3.json",
                APP_DIR / "breast_cancer_news_content.json",
            ]
            for g in guesses:
                if g.exists():
                    articles = load_json_any(g)
                    st.info(f"Using default file: {g.name}")
                    break
            if not articles:
                st.error("No input provided. Upload a file, pick one from the repo root, or enter a valid path.")
        if articles and not isinstance(articles, list):
            st.error("Input JSON must be a list of article dicts.")
            articles = []
    except Exception as e:
        st.exception(e)
        articles = []

# =========================
# PROCESS & DISPLAY
# =========================
results = []
if run_btn and articles:
    with st.spinner("Generating polls…"):
        for idx, art in enumerate(articles, start=1):
            headline = (art.get("headline") or "").strip()
            url = (art.get("url") or "").strip()
            content = (art.get("content") or "").strip()
            polls, facts = build_polls_for_article(headline, url, content)
            results.append({"headline": headline, "url": url, "polls": polls})

            with st.expander(f"#{idx}  {headline or '(no headline)'}", expanded=False):
                if url:
                    st.markdown(f"**Source:** [{url}]({url})")
                col1, col2, col3 = st.columns([1.2, 1, 1])
                with col1:
                    st.markdown("**Indication:** " + (facts["indication"] or "—"))
                    st.markdown("**Drug:** " + (facts["drug"] or "—"))
                    st.markdown("**Phase:** " + (facts["phase"] or "—"))
                with col2:
                    st.markdown("**Publisher:** " + (facts["publisher"] or "—"))
                    st.markdown("**Companies:** " + (", ".join(facts["companies"]) or "—"))
                with col3:
                    ts = [k for k, v in facts["topics"].items() if v]
                    st.markdown("**Topics:** " + (", ".join(ts) if ts else "—"))
                if content:
                    preview = content.strip()
                    if len(preview) > 600:
                        preview = preview[:600] + "…"
                    st.markdown("> " + preview.replace("\n", " "))
                st.markdown("---")
                st.subheader("Generated Polls")
                if not polls:
                    st.info("No grounded polls were generated for this article.")
                else:
                    for i, poll in enumerate(polls, start=1):
                        lines = [ln for ln in poll.splitlines() if ln.strip()]
                        if not lines:
                            continue
                        q = lines[0].replace("Q:", "").strip()
                        opts = [ln[2:].strip() for ln in lines[1:] if ln.startswith("- ")]
                        with st.container():
                            st.markdown(f"**Q{i}. {q}**")
                            st.radio(label=" ", options=opts or ["(no options)"], index=None, key=f"{idx}-{i}", disabled=True)
                        st.markdown("")

    st.markdown("---")
    st.subheader("⬇️ Download")
    out_json = json.dumps(results, ensure_ascii=False, indent=2)
    st.download_button(
        "Download results as JSON",
        data=out_json.encode("utf-8"),
        file_name="twitter_polls_generated.json",
        mime="application/json"
    )

st.markdown("---")
st.caption(
    "Tip: If polls look too generic, increase passes or switch to the larger model. "
    "Each question must mention at least one extracted fact (drug/company/indication/phase/topic) and focus on practice impact."
)
