# app.py
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from urllib.parse import urlparse

import numpy as np
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer
from huggingface_hub import login as hf_login, InferenceClient  # optional (HF path)

# =========================
# DEFAULTS / CONSTANTS
# =========================
LLAMA_MODEL = "meta-llama/llama-3-3-70b-instruct"      # chat_model on watsonx
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"    # optional alt
MAX_TWEET_CHARS = 280
MAX_POLL_OPTIONS = 4
MAX_POLLS_PER_ARTICLE = 4  # <-- hard cap (includes awareness, if present)

st.set_page_config(page_title="Breast Cancer News → Poll Generator", layout="wide")

# =========================
# SIDEBAR / SETTINGS
# =========================
with st.sidebar:
    st.title("⚙️ Settings")

    provider = st.selectbox(
        "Provider",
        ["watsonx", "huggingface"],
        index=0,
        help="Pick the platform to call the model."
    )

    model_name = st.selectbox(
        "Model",
        [LLAMA_MODEL, MISTRAL_MODEL],
        index=0,
        help="Llama 3.3 70B Instruct (watsonx chat_model) or Mistral 7B Instruct."
    )

    if provider == "watsonx":
        wxa_api_key = st.text_input("watsonx API key", type="password")
        wxa_url = st.text_input("watsonx URL", value="https://us-south.ml.cloud.ibm.com")
        wxa_project_or_space = st.text_input(
            "watsonx Project ID (or Space ID)",
            help="If a Space, you can enter either the raw space id or 'space:<id>'."
        )
        use_hf_inference = False
        hf_token = None
    else:
        default_token = st.secrets.get("HF_TOKEN", "")
        hf_token = st.text_input("Hugging Face token", value=default_token, type="password",
                                 help="Fine-grained token with 'Make calls to Inference Providers'.")
        use_hf_inference = st.toggle("Use HF Inference API (recommended)", value=True)

    passes_per_article = st.slider("Generation passes per article", 1, 6, 3)
    temperature = st.slider("Temperature (creativity)", 0.0, 1.2, 0.8, 0.1)
    top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.9, 0.05)

    max_new_tokens = st.slider(
        "Max new tokens per pass",
        min_value=128, max_value=1024, value=480, step=32,
        help="Raise this if polls are getting cut off mid-sentence."
    )

    similarity_dup_threshold = st.slider(
        "Uniqueness threshold (semantic de-dup)", 0.85, 0.99, 0.95, 0.01,
        help="Higher = only keep very distinct polls (per article)."
    )

    allow_rule_fallback = st.toggle(
        "Allow rule-based fallback if LLM fails",
        value=False,
        help="Turn ON only if you're okay with generic polls when the model is unavailable."
    )

    st.markdown("---")
    st.caption("Upload a JSON list like: `[{'headline','url','content'}, ...]`")

# Authenticate HF (if used)
if provider == "huggingface" and hf_token:
    try:
        hf_login(token=hf_token)
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    except Exception as e:
        st.warning(f"Could not authenticate to Hugging Face: {e}")

# =========================
# CACHING HEAVY OBJECTS
# =========================
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_hf_inference_client(model_id: str, token: Optional[str]):
    if not token:
        raise RuntimeError("No HF token provided for Inference API.")
    return InferenceClient(model=model_id, token=token, timeout=180)

@st.cache_resource(show_spinner=False)
def get_local_mistral_pipeline(_model_name: str, token: Optional[str]):
    is_mistral = "mistral" in _model_name.lower()
    if not is_mistral:
        raise RuntimeError("Local mode only supports Mistral in this app.")
    cuda = torch.cuda.is_available()
    st.info("🔧 Loading Mistral locally (needs ~14GB+ VRAM).")
    dtype = torch.float16 if cuda else torch.float32
    tok = AutoTokenizer.from_pretrained(_model_name, token=token, trust_remote_code=True)
    pipe = pipeline(
        task="text-generation",
        model=_model_name,
        tokenizer=tok,
        torch_dtype=dtype,
        device=0 if cuda else -1,
        trust_remote_code=True
    )
    return pipe

# ---- watsonx helper (feature-detect chat vs text) ----
@st.cache_resource(show_spinner=False)
def get_watsonx_model(model_id: str, api_key: str, url: str, project_or_space_id: str, params: dict):
    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import Model
    except ModuleNotFoundError:
        raise RuntimeError(
            "ibm-watsonx-ai SDK is not installed. Add 'ibm-watsonx-ai' to requirements.txt and redeploy."
        )

    creds = Credentials(url=url, api_key=api_key)

    kwargs = {}
    token = (project_or_space_id or "").strip()
    if token.lower().startswith(("space:", "spaces:")):
        kwargs["space_id"] = token.split(":", 1)[-1]
    else:
        kwargs["project_id"] = token

    mdl = Model(model_id=model_id, credentials=creds, params=params, **kwargs)
    return mdl

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
# ENTITY/FACT EXTRACTION (simple)
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
# AWARENESS POLL
# =========================
def build_awareness_poll(headline: str, content: str, url: str) -> str:
    facts = extract_facts(headline, content, url)
    drug, ind = facts["drug"], facts["indication"]

    if drug and ind:
        q = f"Q: Were you aware of this {drug} development for {ind} before reading?"
    elif ind:
        q = f"Q: Were you aware of this {ind} development before reading?"
    else:
        q = f"Q: Were you aware of this development before reading?"

    poll = q + "\n- Yes, was aware\n- No, new information\n- Somewhat aware\n- Will research further"
    poll = enforce_neutrality(poll)
    poll = trim_to_limit(poll, MAX_TWEET_CHARS)
    return poll

# =========================
# LLM MINER
# =========================
class PollMiner:
    def __init__(self, _model_name: str, provider: str, use_hf_inference: bool,
                 hf_token: Optional[str],
                 wxa_api_key: Optional[str], wxa_url: Optional[str], wxa_project_or_space: Optional[str]):
        self.model_name = _model_name
        self.provider = provider
        self.is_mistral = "mistral" in _model_name.lower()

        self.hf_client = None
        self.gen_pipeline = None
        self.wxa_model = None

        try:
            if provider == "watsonx":
                if not (wxa_api_key and wxa_url and wxa_project_or_space):
                    raise RuntimeError("Missing watsonx credentials (API key, URL, Project/Space ID).")

                base_params = {
                    "decoding_method": "sample",
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "max_new_tokens": 480,
                }
                self.wxa_model = get_watsonx_model(_model_name, wxa_api_key, wxa_url, wxa_project_or_space, base_params)

            elif provider == "huggingface":
                if use_hf_inference:
                    self.hf_client = get_hf_inference_client(_model_name, hf_token)
                else:
                    if not self.is_mistral:
                        raise RuntimeError("Local mode only implemented for Mistral.")
                    self.gen_pipeline = get_local_mistral_pipeline(_model_name, hf_token)
            else:
                raise RuntimeError(f"Unsupported provider: {provider}")

        except Exception as e:
            st.error(f"❌ Model init failed: {e}")
            self.hf_client = None
            self.gen_pipeline = None
            self.wxa_model = None

    # ---------- HCP-perspective, content-bound prompt (max 4 polls) ----------
    def _format_prompt(self, headline: str, url: str, content: str, facts: Dict) -> str:
        snippet = content.strip()
        if len(snippet) > 1600:
            snippet = snippet[:1600]
        facts_str = json.dumps({
            "drug": facts["drug"],
            "companies": facts["companies"][:3],
            "indication": facts["indication"],
            "phase": facts["phase"],
            "topics_true": [k for k, v in facts["topics"].items() if v]
        }, ensure_ascii=False)

        return f"""
You are an assistant that writes Twitter/X poll questions to understand a **doctor's (HCP) perspective** on ONE breast cancer news article.

STRICT CONTENT RULES — READ CAREFULLY:
- USE ONLY information that appears in the headline or article content below (or in Key Facts). Do NOT invent drugs, companies, endpoints, biomarkers, or settings.
- Each poll MUST clearly reference the specific context present in the article (e.g., drug name, indication, phase/setting, relevant endpoint if mentioned).
- Produce AT MOST **4** polls and make them mutually DISTINCT in focus (avoid paraphrases):
  • practice impact (practice-informing vs practice-changing)  
  • intent to use in practice  
  • patient selection / setting (line of therapy, subtype, biomarker)  
  • endpoints / treatment discussions (e.g., PFS/OS/ORR if mentioned)
- Keep each poll UNDER 280 characters total (question + options) and include 3–4 concise options.
- Return COMPLETE polls only (no truncated options).
- Purpose: elicit the **doctor's perspective** (practice intent), not generic Q&A.

ARTICLE INFORMATION:
Headline: "{headline}"
URL: {url}
Key Facts (only for grounding; do not add new facts): {facts_str}

ARTICLE CONTENT (truncated):
{snippet}

OUTPUT FORMAT (up to 4 polls; no extra text):
Q: <HCP-perspective question that strictly reflects article content>
- Option 1
- Option 2
- Option 3
- Option 4 (optional)
"""

    # ----- Provider-specific calls -----
    def _gen_watsonx(self, prompt: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
        if not self.wxa_model:
            raise RuntimeError("watsonx model not initialized")

        try:
            self.wxa_model.params.update({
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_new_tokens": int(max_new_tokens),
                "decoding_method": "sample",
            })
        except Exception:
            pass

        try:
            if hasattr(self.wxa_model, "start_chat"):
                chat = self.wxa_model.start_chat()
                resp = chat.send_message(prompt)
                text = getattr(resp, "message", None) or getattr(resp, "generated_text", None)
                if isinstance(text, str) and text.strip():
                    return text
                if isinstance(resp, dict):
                    return resp.get("generated_text") or resp.get("message") or json.dumps(resp)
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

    def _gen_hf(self, prompt: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
        chat = getattr(self.hf_client, "chat", None)
        if chat and hasattr(chat, "completions") and hasattr(chat.completions, "create"):
            resp = chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            choice = resp.choices[0]
            if hasattr(choice, "message") and choice.message and "content" in choice.message:
                return choice.message["content"]
            if hasattr(choice, "text"):
                return choice.text

        return self.hf_client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.05,
            return_full_text=False,
            stream=False,
        )

    # --- uniqueness-only filtering + diversity buckets ---
    @staticmethod
    def _stem(line: str) -> str:
        s = line.lower()
        s = re.sub(r"^q\s*:\s*", "", s)
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        s = re.sub(r"\b(the|a|an|for|of|to|in|on|with|about|is|are|this|that|how|will|your|does|do)\b", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _bucket(q_text: str) -> str:
        t = q_text.lower()
        if "aware" in t or "awareness" in t:
            return "awareness"
        if ("practice-changing" in t) or ("practice changing" in t) or ("practice-informing" in t) or ("impact" in t):
            return "practice_impact"
        if ("will you use" in t) or ("use this" in t) or ("in your practice" in t) or ("consider using" in t):
            return "intent_to_use"
        if any(k in t for k in ["which patients", "patient", "line of therapy", "first-line", "adjuvant", "setting", "subtype", "biomarker"]):
            return "patient_selection"
        if any(k in t for k in ["pfs", "os", "orr", "endpoint", "discussions"]):
            return "endpoints"
        return "other"

    def _select_diverse_top4(self, polls: List[str], embedder) -> List[str]:
        """Pick at most one per bucket in priority order, max 4 total."""
        if not polls:
            return []

        # exact de-dup
        uniq = list(dict.fromkeys([normalize_ws(p) for p in polls]))
        # semantic de-dup
        uniq = dedup_semantic(uniq, similarity_dup_threshold, embedder)

        # stem de-dup within selection loop
        priority = ["practice_impact", "intent_to_use", "patient_selection", "endpoints", "other"]
        chosen: List[str] = []
        seen_stems: List[str] = []

        # bucketize
        by_bucket: Dict[str, List[str]] = {}
        for p in uniq:
            qline = p.splitlines()[0] if p else ""
            bucket = self._bucket(qline)
            by_bucket.setdefault(bucket, []).append(p)

        for b in priority:
            for candidate in by_bucket.get(b, []):
                stem = self._stem(candidate.splitlines()[0])
                if not seen_stems:
                    chosen.append(candidate); seen_stems.append(stem)
                    break
                # stem cosine to avoid paraphrases
                if util.cos_sim(embedder.encode(stem), embedder.encode(seen_stems)).max().item() < 0.90:
                    chosen.append(candidate); seen_stems.append(stem)
                    break
                if len(chosen) >= MAX_POLLS_PER_ARTICLE:
                    break
            if len(chosen) >= MAX_POLLS_PER_ARTICLE:
                break

        # If still under cap, fill with remaining unique (non-paraphrase)
        if len(chosen) < MAX_POLLS_PER_ARTICLE:
            for p in uniq:
                if p in chosen:
                    continue
                stem = self._stem(p.splitlines()[0])
                if util.cos_sim(embedder.encode(stem), embedder.encode(seen_stems)).max().item() < 0.90:
                    chosen.append(p); seen_stems.append(stem)
                if len(chosen) >= MAX_POLLS_PER_ARTICLE:
                    break

        return chosen[:MAX_POLLS_PER_ARTICLE]

    def generate(self, headline: str, url: str, content: str, facts: Dict,
                 passes: int, temperature: float, top_p: float, max_new_tokens: int) -> List[str]:

        ready = any([self.wxa_model, self.hf_client, self.gen_pipeline])
        if not ready:
            st.warning("Model not initialized; using rule-based fallback.")
            return []

        all_blocks: List[str] = []

        # Per-article jitter to reduce sameness
        seed_jitter = (abs(hash(headline + url)) % 1000) / 1000.0
        jitter_temp = min(1.2, max(0.2, temperature + (seed_jitter - 0.5) * 0.15))
        jitter_top_p = min(1.0, max(0.6, top_p + (seed_jitter - 0.5) * 0.15))

        for i in range(passes):
            try:
                with st.spinner(f"Generating polls (pass {i+1}/{passes})..."):
                    prompt = self._format_prompt(headline, url, content, facts)

                    if self.provider == "watsonx":
                        out = self._gen_watsonx(prompt, jitter_temp, jitter_top_p, max_new_tokens)
                        st.caption("✅ Response via watsonx.ai")
                    elif self.hf_client:
                        out = self._gen_hf(prompt, jitter_temp, jitter_top_p, max_new_tokens)
                        st.caption("✅ Response via HF Inference API")
                    else:
                        result = self.gen_pipeline(
                            prompt,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=jitter_temp,
                            top_p=jitter_top_p,
                            num_return_sequences=1,
                            pad_token_id=self.gen_pipeline.tokenizer.eos_token_id,
                            return_full_text=False,
                        )
                        out = result[0]["generated_text"]
                        st.caption("✅ Response via local Mistral")

                    blocks = parse_polls_block(out)
                    for b in blocks:
                        b = enforce_neutrality(b)
                        b = trim_to_limit(b, MAX_TWEET_CHARS)
                        all_blocks.append(b)

            except Exception as e:
                st.warning(f"Generation attempt {i+1} failed: {e}")
                continue

        # Keep only polls that look valid
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

        embedder = get_embedder()
        return self._select_diverse_top4(valid, embedder)

# =========================
# RULE-BASED FALLBACK (kept simple)
# =========================
def rule_based_polls(facts: Dict) -> List[str]:
    polls: List[str] = []
    def add(q, options):
        block = "Q: " + q.strip()
        opts = ["- " + o.strip() for o in options if o.strip()]
        if len(opts) >= 3:
            block = enforce_neutrality(block + "\n" + "\n".join(opts))
            block = trim_to_limit(block, MAX_TWEET_CHARS)
            polls.append(block)

    drug, ind, phase = facts["drug"], facts["indication"], facts["phase"]
    tp = facts["topics"]

    if drug and ind:
        add(
            f"Is this information about {drug} for {ind} practice-changing for you?",
            ["Practice-changing", "Practice-informing", "No impact", "Need more data"]
        )
        add(
            f"Will you use {drug} for {ind} in your practice?",
            ["Yes, broadly", "Yes, select patients", "Not now", "Awaiting guidelines"]
        )
        add(
            f"For which {ind} patients would you consider {drug}?",
            ["First-line", "Later-line", "Specific biomarker", "Not applicable"]
        )
        add(
            f"Do these results change your treatment discussions?",
            ["Significantly", "Somewhat", "Not really", "Unsure"]
        )

    return polls[:MAX_POLLS_PER_ARTICLE]

# =========================
# PIPELINE (one article)
# =========================
def build_polls_for_article(headline: str, url: str, content: str):
    facts = extract_facts(headline, content, url)
    awareness = build_awareness_poll(headline, content, url)

    miner = PollMiner(
        model_name, provider, use_hf_inference,
        hf_token,
        wxa_api_key if provider == "watsonx" else None,
        wxa_url if provider == "watsonx" else None,
        wxa_project_or_space if provider == "watsonx" else None
    )

    llm_polls = miner.generate(headline, url, content, facts,
                               passes_per_article, temperature, top_p, max_new_tokens)

    if not llm_polls and allow_rule_fallback:
        llm_polls = rule_based_polls(facts)

    # Combine with awareness and cap to MAX_POLLS_PER_ARTICLE with diversity
    embedder = get_embedder()
    all_polls = [awareness] + llm_polls
    # exact dedup -> semantic -> stem diversity
    all_polls = list(dict.fromkeys([normalize_ws(p) for p in all_polls]))
    all_polls = dedup_semantic(all_polls, max(similarity_dup_threshold, 0.93), embedder)

    # simple stem dedup and cap
    stems = []
    final = []
    for p in all_polls:
        if len(final) >= MAX_POLLS_PER_ARTICLE:
            break
        stem = PollMiner._stem(p.splitlines()[0])
        if not stems:
            final.append(p); stems.append(stem); continue
        if util.cos_sim(embedder.encode(stem), embedder.encode(stems)).max().item() < 0.90:
            final.append(p); stems.append(stem)

    return final, facts

# =========================
# FILE INPUTS
# =========================
st.header("📰 Breast Cancer News → 🗳️ Poll Generator")

uploaded = st.file_uploader("Upload JSON (list of objects with 'headline','url','content')", type=["json"])

APP_DIR = Path(__file__).parent.resolve()
repo_files: List[Path] = sorted([p for p in APP_DIR.glob("*.json") if p.is_file()])
repo_choices = ["(none)"] + [p.name for p in repo_files]
repo_pick = st.selectbox("…or pick a JSON file from this repo (root)", options=repo_choices, index=0)

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
                p = APP_DIR / p
            if not p.exists():
                st.error(f"Path not found: {p}")
            else:
                articles = load_json_any(p)
        else:
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
    if provider == "huggingface" and use_hf_inference and not hf_token:
        st.error("You selected HF Inference API but no token is set.")
    elif provider == "watsonx" and not (wxa_api_key and wxa_url and wxa_project_or_space):
        st.error("watsonx selected but credentials are missing.")
    else:
        with st.spinner("Generating polls…"):
            for idx, art in enumerate(articles, start=1):
                headline = (art.get("headline") or "").strip()
                url = (art.get("url") or "").strip()
                content = (art.get("content") or "").strip()
                try:
                    polls, facts = build_polls_for_article(headline, url, content)
                except Exception as e:
                    st.error(f"#{idx} {headline or '(no headline)'}: {e}")
                    continue

                # ---- Optional comment input per article ----
                comment_key = f"comment-{idx}"
                st.session_state.setdefault(comment_key, "")
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
                    st.subheader("Generated Polls (max 4)")
                    if not polls:
                        st.info("No unique polls were generated for this article.")
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
                    st.text_area("Optional HCP comment (will be exported)", key=comment_key, height=100)

                # add to results including comment
                results.append({
                    "headline": headline,
                    "url": url,
                    "polls": polls,
                    "comment": st.session_state.get(comment_key, "")
                })

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
    "If polls look truncated, raise 'Max new tokens per pass'. "
    "For watsonx, provide API key, instance URL, and Project/Space ID. "
    "For HF, ensure your token allows 'Make calls to Inference Providers'."
)
