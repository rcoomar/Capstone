# app.py
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
from datetime import datetime

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

# Lock to server-side creds by default
DEFAULT_PROVIDER = "watsonx"
DEFAULT_MODEL = LLAMA_MODEL

st.set_page_config(page_title="Breast Cancer News → Poll Generator", layout="wide")

# =========================
# Secrets / Env helpers
# =========================
def get_watsonx_creds():
    api_key = st.secrets.get("WATSONX_API_KEY") or os.getenv("WATSONX_API_KEY", "")
    url = st.secrets.get("WATSONX_URL") or os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    space_id = st.secrets.get("WATSONX_SPACE_ID") or os.getenv("WATSONX_SPACE_ID", "")
    proj_id  = st.secrets.get("WATSONX_PROJECT_ID") or os.getenv("WATSONX_PROJECT_ID", "")
    space_or_project = space_id or proj_id
    return api_key, url, space_or_project

def get_hf_token():
    return st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN", "")

# =========================
# SIDEBAR / SETTINGS
# =========================
with st.sidebar:
    st.title("⚙️ Settings")

    provider = st.selectbox(
        "Provider",
        ["watsonx", "huggingface"],
        index=0 if DEFAULT_PROVIDER == "watsonx" else 1,
    )

    model_name = st.selectbox(
        "Model",
        [LLAMA_MODEL, MISTRAL_MODEL],
        index=0 if DEFAULT_MODEL == LLAMA_MODEL else 1,
        help="Llama 3.3 70B Instruct (watsonx chat_model) or Mistral 7B Instruct.",
    )

    if provider == "watsonx":
        wxa_api_key, wxa_url, wxa_project_or_space = get_watsonx_creds()
        st.caption("Using server-side watsonx credentials (Streamlit secrets / env).")
        use_hf_inference = False
        hf_token = None
    else:
        hf_token = get_hf_token()
        use_hf_inference = st.toggle("Use HF Inference API (recommended)", value=True)
        if not hf_token:
            st.warning("No HF_TOKEN found in secrets/env. Hugging Face calls will fail without it.")

    passes_per_article = st.slider("Generation passes per article", 1, 6, 3)
    temperature = st.slider("Temperature (creativity)", 0.0, 1.2, 0.8, 0.1)
    top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.9, 0.05)

    max_new_tokens = st.slider(
        "Max new tokens per pass",
        min_value=128, max_value=1024, value=480, step=32,
        help="Raise this if polls are getting cut off mid-sentence.",
    )

    similarity_dup_threshold = st.slider(
        "Uniqueness threshold (semantic de-dup)", 0.85, 0.99, 0.95, 0.01,
        help="Higher = only keep very distinct polls (per article).",
    )

    allow_rule_fallback = st.toggle(
        "Allow rule-based fallback if LLM fails",
        value=False,
        help="Turn ON only if you're okay with generic polls when the model is unavailable.",
    )

    st.markdown("---")
    st.subheader("Filter by date (Published)")
    # Month & Year filter from published_date; we apply when processing
    month_choice = st.selectbox(
        "Month",
        ["All", "January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"],
        index=0
    )
    year_choice = st.text_input("Year (YYYY or blank for All)", value="")

    st.markdown("---")
    st.subheader("Grounding validation")
    grounding_threshold = st.slider(
        "Flag threshold (overall grounding)",
        0.50, 0.99, 0.75, 0.01,
        help="Questions scoring below this will be flagged for human review."
    )
    entity_weight = st.slider(
        "Entity weight in overall score",
        0.0, 1.0, 0.30, 0.05,
        help="Overall = (1 - weight) * semantic_similarity + weight * entity_overlap"
    )

# Authenticate HF (server-side only)
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
        trust_remote_code=True,
    )
    return pipe

# ---- watsonx helper
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
# JSON → FACTS (NO inference)
# =========================
def company_from_url(url: str) -> Optional[str]:
    try:
        host = urlparse(url).netloc.lower()
        if not host:
            return None
        for prefix in ("www.", "amp.", "m.", "news."):
            if host.startswith(prefix):
                host = host[len(prefix):]
        parts = host.split(".")
        brand = parts[-2] if len(parts) >= 2 else parts[0]
        brand = brand.replace("-", " ").strip()
        if not brand:
            return None
        return brand[0].upper() + brand[1:]
    except Exception:
        return None

def _coerce_list(x) -> List[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    return [str(x).strip()]

def parse_date_safe(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    fmts = [
        "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
        "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S", "%b %d, %Y", "%B %d, %Y",
        "%Y-%m", "%B %Y", "%b %Y"
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f)
        except Exception:
            continue
    return None

def adapt_article_schema(art: Dict) -> Dict:
    headline = (art.get("headline") or art.get("title") or "").strip()
    url = (art.get("url") or "").strip()
    content = (art.get("content") or art.get("article") or "").strip()
    summary = (art.get("summary_280") or art.get("summary_tweet") or art.get("summary") or "").strip()
    published_date_raw = (art.get("published_date") or art.get("date") or "").strip()

    entities = art.get("entities") or {}
    drugs = _coerce_list(entities.get("drug_names") or art.get("drugs"))
    companies = _coerce_list(entities.get("company_name") or art.get("companies"))
    trial_names = _coerce_list(entities.get("trial_names"))
    trial_phases = _coerce_list(entities.get("trial_phases"))
    indications = _coerce_list(entities.get("indications"))

    # publisher derived from URL only (no inference)
    publisher = company_from_url(url)

    # grounding text = content; if missing, use headline + summary
    grounding_text = content if content else (headline + "\n\n" + summary)

    return {
        "headline": headline,
        "url": url,
        "content": content,
        "summary": summary,
        "published_date": published_date_raw,
        "published_dt": parse_date_safe(published_date_raw),
        "entities": {
            "drug_names": drugs,
            "company_name": companies,
            "trial_names": trial_names,
            "trial_phases": trial_phases,
            "indications": indications,
        },
        "publisher": publisher,
        "grounding_text": grounding_text,
        "raw": art,  # keep original for display
    }

# =========================
# AWARENESS POLL (light-touch, JSON-only)
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
        q = f"Q: Were you aware of this development before reading?"

    poll = q + "\n- Yes, was aware\n- No, new information\n- Somewhat aware\n- Will research further"
    poll = enforce_neutrality(poll)
    poll = trim_to_limit(poll, MAX_TWEET_CHARS)
    return poll

# =========================
# LLM MINER — uses CONTENT only; no entity extraction
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
                base_params = {"decoding_method": "sample", "temperature": 0.8, "top_p": 0.9, "max_new_tokens": 480}
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

    def _format_prompt(self, headline: str, url: str, grounding_text: str, key_facts: Dict) -> str:
        snippet = grounding_text.strip()
        if len(snippet) > 1600:
            snippet = snippet[:1600]

        # Only pass JSON-provided facts. No inferred fields.
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
        if not polls:
            return []
        uniq = list(dict.fromkeys([normalize_ws(p) for p in polls]))
        uniq = dedup_semantic(uniq, similarity_dup_threshold, embedder)

        priority = ["practice_impact", "intent_to_use", "patient_selection", "endpoints", "other"]
        chosen: List[str] = []
        seen_stems: List[str] = []

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
                if util.cos_sim(get_embedder().encode(stem), get_embedder().encode(seen_stems)).max().item() < 0.90:
                    chosen.append(candidate); seen_stems.append(stem)
                    break
                if len(chosen) >= MAX_POLLS_PER_ARTICLE:
                    break
            if len(chosen) >= MAX_POLLS_PER_ARTICLE:
                break

        if len(chosen) < MAX_POLLS_PER_ARTICLE:
            for p in uniq:
                if p in chosen:
                    continue
                stem = self._stem(p.splitlines()[0])
                if util.cos_sim(get_embedder().encode(stem), get_embedder().encode(seen_stems)).max().item() < 0.90:
                    chosen.append(p); seen_stems.append(stem)
                if len(chosen) >= MAX_POLLS_PER_ARTICLE:
                    break

        return chosen[:MAX_POLLS_PER_ARTICLE]

    def generate(self, headline: str, url: str, grounding_text: str, key_facts: Dict,
                 passes: int, temperature: float, top_p: float, max_new_tokens: int) -> List[str]:

        ready = any([self.wxa_model, self.hf_client, self.gen_pipeline])
        if not ready:
            st.warning("Model not initialized; using rule-based fallback.")
            return []

        all_blocks: List[str] = []

        seed_jitter = (abs(hash(headline + url)) % 1000) / 1000.0
        jitter_temp = min(1.2, max(0.2, temperature + (seed_jitter - 0.5) * 0.15))
        jitter_top_p = min(1.0, max(0.6, top_p + (seed_jitter - 0.5) * 0.15))

        for i in range(passes):
            try:
                with st.spinner(f"Generating polls (pass {i+1}/{passes})..."):
                    prompt = self._format_prompt(headline, url, grounding_text, key_facts)

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
# GROUNDING VALIDATION HELPERS
# =========================
def _chunk_text_for_similarity(text: str, chunk_size: int = 480, overlap: int = 120) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

def _normalize(txt: str) -> str:
    return re.sub(r"\s+", " ", (txt or "").strip().lower())

def _contains_phrase(q: str, phrase: str) -> bool:
    qn = _normalize(q)
    pn = _normalize(phrase)
    return pn and pn in qn

def _entity_overlap_score(question_text: str, json_entities: Dict) -> float:
    """
    Weighted presence of key entities in the QUESTION (not options), using JSON only:
      - drug_names (0.5)
      - indications (0.3)
      - trial_phases (0.2)
    """
    q = _normalize(question_text)
    w_drug, w_ind, w_ph = 0.5, 0.3, 0.2
    total_w = 0.0
    score = 0.0

    drugs = [d for d in (json_entities.get("drug_names") or []) if isinstance(d, str) and d.strip()]
    if drugs:
        total_w += w_drug
        if any(_contains_phrase(q, d) for d in drugs):
            score += w_drug

    inds = [i for i in (json_entities.get("indications") or []) if isinstance(i, str) and i.strip()]
    if inds:
        total_w += w_ind
        # Allow partial token overlap (e.g., "TNBC" / "triple-negative breast cancer")
        present = any(_contains_phrase(q, ind) or "tnbc" in q for ind in inds)
        if present:
            score += w_ind

    phases = [p for p in (json_entities.get("trial_phases") or []) if isinstance(p, str) and p.strip()]
    if phases:
        total_w += w_ph
        present = any(_contains_phrase(q, p) or _contains_phrase(q, p.replace("Phase ", "")) for p in phases)
        if present:
            score += w_ph

    if total_w == 0.0:
        return 0.5
    return score / total_w

def grounding_semantic_similarity(question_text: str, grounding_text: str, embedder) -> float:
    q = question_text.strip()
    if not q or not grounding_text:
        return 0.0
    q_emb = embedder.encode([q], normalize_embeddings=True)
    chunks = _chunk_text_for_similarity(grounding_text, chunk_size=480, overlap=120)
    if not chunks:
        return 0.0
    c_embs = embedder.encode(chunks, normalize_embeddings=True)
    sims = util.cos_sim(q_emb, c_embs).cpu().numpy()[0]
    return float(np.max(sims))

def grounding_overall_score(question_text: str, grounding_text: str, json_entities: Dict, embedder, entity_weight: float) -> Dict[str, float]:
    sem = grounding_semantic_similarity(question_text, grounding_text, embedder)  # 0..1
    ent = _entity_overlap_score(question_text, json_entities)                     # 0..1
    overall = (1.0 - float(entity_weight)) * sem + float(entity_weight) * ent
    return {"semantic": sem, "entity": ent, "overall": overall}

# =========================
# PIPELINE (per article)
# =========================
def build_polls_for_article(adapted: Dict):
    headline = adapted["headline"]
    url = adapted["url"]
    grounding_text = adapted["grounding_text"]
    summary = adapted["summary"]
    json_entities = adapted["entities"]

    # Awareness poll based only on JSON entities
    awareness = build_awareness_poll(json_entities)

    # Model init
    if provider == "watsonx":
        wxa_api_key, wxa_url, wxa_project_or_space = get_watsonx_creds()
        miner = PollMiner(
            model_name, provider, False,
            None,
            wxa_api_key, wxa_url, wxa_project_or_space
        )
    else:
        miner = PollMiner(
            model_name, provider, True,
            get_hf_token(),
            None, None, None
        )

    llm_polls = miner.generate(
        headline, url, grounding_text, json_entities,
        passes_per_article, temperature, top_p, max_new_tokens
    )

    # Optional fallback
    if not llm_polls and allow_rule_fallback:
        llm_polls = [awareness]  # keep simple if model is unavailable

    embedder = get_embedder()
    all_polls = [awareness] + llm_polls
    all_polls = list(dict.fromkeys([normalize_ws(p) for p in all_polls]))
    all_polls = dedup_semantic(all_polls, max(similarity_dup_threshold, 0.93), embedder)

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

    # Scores
    poll_scores: List[Dict] = []
    for poll in final:
        lines = [ln for ln in poll.splitlines() if ln.strip()]
        q_line = lines[0]
        question_text = re.sub(r"^Q\s*:\s*", "", q_line, flags=re.I).strip()
        scores = grounding_overall_score(question_text, grounding_text, json_entities, embedder, entity_weight)
        poll_scores.append(scores)

    return final, poll_scores

# =========================
# FILE INPUTS
# =========================
st.header("📰 Breast Cancer News → 🗳️ Poll Generator")

uploaded = st.file_uploader("Upload JSON", type=["json"])

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

def passes_date_filter(published_dt: Optional[datetime], month_choice: str, year_choice: str) -> bool:
    # If no filters set, include all
    if (month_choice == "All") and (not year_choice.strip()):
        return True
    # If no date parse, include only when filters are "All"
    if not published_dt:
        return False
    m_ok = True
    y_ok = True
    if month_choice != "All":
        month_map = {
            "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
            "July":7,"August":8,"September":9,"October":10,"November":11,"December":12
        }
        m_ok = (published_dt.month == month_map[month_choice])
    if year_choice.strip():
        try:
            y = int(year_choice.strip())
            y_ok = (published_dt.year == y)
        except Exception:
            y_ok = True  # ignore bad input
    return m_ok and y_ok

articles_raw: List[Dict] = []
if run_btn:
    try:
        if uploaded is not None:
            articles_raw = json.load(uploaded)
        elif repo_pick != "(none)":
            chosen = APP_DIR / repo_pick
            articles_raw = load_json_any(chosen)
        elif path_input.strip():
            p = Path(path_input)
            if not p.is_absolute():
                p = APP_DIR / p
            if not p.exists():
                st.error(f"Path not found: {p}")
            else:
                articles_raw = load_json_any(p)
        else:
            st.error("No input provided. Upload a file, pick one from the repo root, or enter a valid path.")
        if articles_raw and not isinstance(articles_raw, list):
            st.error("Input JSON must be a list of article dicts.")
            articles_raw = []
    except Exception as e:
        st.exception(e)
        articles_raw = []

# =========================
# PROCESS & DISPLAY
# =========================
results = []
if run_btn and articles_raw:
    # Adapt and filter by Month/Year
    adapted_all: List[Dict] = [adapt_article_schema(art) for art in articles_raw]
    adapted = [a for a in adapted_all if passes_date_filter(a["published_dt"], month_choice, year_choice)]

    if not adapted:
        st.info("No articles matched the Month/Year filters.")
    else:
        if provider == "huggingface" and use_hf_inference and not get_hf_token():
            st.error("Hugging Face selected but HF_TOKEN is missing from secrets/env.")
        elif provider == "watsonx":
            _api, _url, _sp = get_watsonx_creds()
            if not (_api and _url and _sp):
                st.error("watsonx selected but server-side credentials are missing (WATSONX_API_KEY/URL and Space or Project ID).")
            else:
                with st.spinner("Generating polls…"):
                    for idx, a in enumerate(adapted, start=1):
                        try:
                            polls, poll_scores = build_polls_for_article(a)
                        except Exception as e:
                            st.error(f"#{idx} {a['headline'] or '(no headline)'}: {e}")
                            continue

                        # UI — no headline truncation: print full headline inside expander
                        comment_key = f"comment-{idx}"
                        st.session_state.setdefault(comment_key, "")
                        with st.expander(f"#{idx}  (open to view)", expanded=False):
                            st.markdown(f"### 📰 Headline\n{a['headline'] or '—'}")
                            # Source + Published
                            src_line = []
                            if a["url"]:
                                src_line.append(f"**Source:** [{a['url']}]({a['url']})")
                            if a["published_date"]:
                                src_line.append(f"**Published:** {a['published_date']}")
                            if src_line:
                                st.markdown(" &nbsp; • &nbsp; ".join(src_line))

                            # Curated summary panel
                            col1, col2, col3 = st.columns([1.4, 1.2, 1.2])
                            ents = a["entities"]
                            with col1:
                                st.markdown("**Drugs/Products:** " + (", ".join(ents.get("drug_names") or []) or "—"))
                                st.markdown("**Indications:** " + (", ".join(ents.get("indications") or []) or "—"))
                            with col2:
                                st.markdown("**Companies:** " + (", ".join(ents.get("company_name") or []) or "—"))
                                st.markdown("**Trial phases:** " + (", ".join(ents.get("trial_phases") or []) or "—"))
                            with col3:
                                st.markdown("**Trials:** " + (", ".join(ents.get("trial_names") or []) or "—"))
                                st.markdown("**Publisher (from URL):** " + (a.get("publisher") or "—"))

                            # Summary / Content preview (no truncation of headline)
                            if a["summary"]:
                                st.markdown("**Summary:**")
                                st.markdown("> " + a["summary"].replace("\n", " "))
                            elif a["content"]:
                                preview = a["content"].strip()
                                if len(preview) > 600:
                                    preview = preview[:600] + "…"
                                st.markdown("> " + preview.replace("\n", " "))

                            st.markdown("---")
                            st.subheader("Generated Polls (max 4)")
                            if not polls:
                                st.info("No unique polls were generated for this article.")
                                selected_answers = {}
                            else:
                                selected_answers = {}
                                for i, poll in enumerate(polls, start=1):
                                    lines = [ln for ln in poll.splitlines() if ln.strip()]
                                    if not lines:
                                        continue
                                    q = lines[0].replace("Q:", "").strip()
                                    opts = [ln[2:].strip() for ln in lines[1:] if ln.startswith("- ")]

                                    sc = poll_scores[i-1] if i-1 < len(poll_scores) else {"semantic": 0.0, "entity": 0.0, "overall": 0.0}
                                    needs_review = sc["overall"] < grounding_threshold
                                    badge = f"Score: **{sc['overall']:.2f}**  (semantic {sc['semantic']:.2f} · entity {sc['entity']:.2f})"
                                    flag = " ⚠️ Needs review" if needs_review else " ✅ Grounded"
                                    st.markdown(f"**Q{i}. {q}**  &nbsp;&nbsp; {badge}{flag}")

                                    with st.container():
                                        answer_key = f"{idx}-{i}"
                                        selected_answer = st.radio(
                                            label=" ",
                                            options=opts or ["(no options)"],
                                            index=None,
                                            key=answer_key
                                        )
                                        selected_answers[f"poll_{i}"] = {
                                            "question": q,
                                            "selected_answer": selected_answer if selected_answer else "Not answered",
                                            "options": opts,
                                            "grounding_score": {
                                                "overall": round(sc["overall"], 4),
                                                "semantic": round(sc["semantic"], 4),
                                                "entity": round(sc["entity"], 4),
                                                "threshold": grounding_threshold,
                                                "needs_review": needs_review,
                                            }
                                        }
                                    st.markdown("")

                            st.markdown("---")
                            st.subheader("Raw JSON (article)")
                            st.json(a["raw"])  # show every field present in input JSON

                            st.text_area("Optional HCP comment (will be exported)", key=comment_key, height=100)

                        results.append({
                            "headline": a["headline"],
                            "url": a["url"],
                            "published_date": a["published_date"],
                            "polls": polls,
                            "selected_answers": selected_answers,
                            "comment": st.session_state.get(comment_key, ""),
                            "entities": a["entities"],
                            "summary": a["summary"],
                        })

                st.markdown("---")
                st.subheader("⬇️ Download")
                out_json = json.dumps(results, ensure_ascii=False, indent=2)
                st.download_button(
                    "Download results as JSON",
                    data=out_json.encode("utf-8"),
                    file_name="twitter_polls_generated.json",
                    mime="application/json",
                )

st.markdown("---")
st.caption(
    "Reads JSON as-is (no entity inference). Questions are generated from content; all original fields are displayed. "
    "Use the Month & Year filters in the sidebar."
)
