# app.py
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from urllib.parse import urlparse

import numpy as np
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer

# Optional (HF path only)
from huggingface_hub import login as hf_login, InferenceClient

# =========================
# DEFAULTS / CONSTANTS
# =========================
LLAMA_MODEL = "meta-llama/llama-3-3-70b-instruct"      # chat_model on watsonx
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"    # optional alt
MAX_TWEET_CHARS = 280
MAX_POLL_OPTIONS = 4

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

    # STRONGER de-dup by default
    similarity_dup_threshold = st.slider(
        "Semantic de-dup threshold", 0.85, 0.99, 0.96, 0.01,
        help="Higher = only keep very distinct polls."
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

    # If the user provided a Space id, prefer space_id; else use project_id
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

CAPS_ALLOW = {
    "FDA","HR+","HER2","HER2+","HER2-","PFS","OS","ORR","DFS","EFS","QoL",
    "CDK4/6","PIK3CA","PD-1","PD-L1","CTLA-4","AI","IHC","NCCN"
}

ORG_KEYWORDS = (
    "pharma|pharmaceutical|pharmaceuticals|therapeutic|therapeutics|bio|biotech|biosciences|"
    "bioscience|biopharma|ltd|inc|corp|corporation|plc|ag|sa|nv|co|gmbh|kk|kkk|limited|srl"
)

KNOWN_PHARMA = {
    "Genentech","Roche","Novartis","Pfizer","AstraZeneca","Eli Lilly","Merck","MSD",
    "Sanofi","Gilead","BMS","Bristol Myers Squibb","Amgen","GSK","Bayer","Takeda",
    "Boehringer Ingelheim","BeiGene","Seagen","Sermonix Pharma","Sermonix","AbbVie","Regeneron",
    "Biogen","Moderna","Vertex","Incyte","Blueprint Medicines","Jazz Pharmaceuticals","Ipsen",
    "Merck KGaA","Daiichi Sankyo","Astellas","Servier","Exact Sciences","Illumina"
}

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
# ENTITY/FACT EXTRACTION (tightened)
# =========================

ORG_SUFFIXES = r"(?: Inc\.?| Corp\.?| Corporation| Ltd\.?| LLC| plc| AG| SA| NV| Co\.?| GmbH| KGaA| KK| Limited| SRL)"
PHARMA_HINTS_RE = re.compile(
    rf"\b([A-Z][A-Za-z&\-]+(?:\s+[A-Z][A-Za-z&\-]+){{0,3}})\s*(?:{ORG_SUFFIXES})?\b",
    flags=re.UNICODE
)

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
            "msd.com": "MSD",
            "sanofi.com": "Sanofi",
            "gilead.com": "Gilead",
            "bms.com": "Bristol Myers Squibb",
            "amgen.com": "Amgen",
            "gsk.com": "GSK",
            "bayer.com": "Bayer",
            "takeda.com": "Takeda",
            "boehringer-ingelheim.com": "Boehringer Ingelheim",
            "beigene.com": "BeiGene",
            "seagen.com": "Seagen",
            "abbvie.com": "AbbVie",
            "regeneron.com": "Regeneron",
            "astellas.com": "Astellas",
            "daiichisankyo.com": "Daiichi Sankyo",
        }
        if host in mapping:
            return mapping[host]
        return None
    except Exception:
        return None

def _looks_like_pharma(name: str) -> bool:
    if name in KNOWN_PHARMA:
        return True
    if re.search(rf"\b({ORG_KEYWORDS})\b", name.lower()):
        return True
    return False

def find_pharma_biotech_companies(text: str, url: str) -> List[str]:
    text_clean = re.sub(r"\s+", " ", text)
    found: Set[str] = set()

    # 1) known pharma names plain
    for k in KNOWN_PHARMA:
        if re.search(rf"\b{k}\b", text_clean, flags=re.I):
            found.add(k)

    # 2) generic org pattern + pharma keywords
    for m in PHARMA_HINTS_RE.finditer(text_clean):
        cand = m.group(0).strip()
        if _looks_like_pharma(cand):
            found.add(re.sub(r"\s+", " ", cand))

    # 3) simple context cues: "by/ from <Company>"
    for m in re.finditer(r"\b(?:by|from)\s+([A-Z][A-Za-z&\-\s]{2,60})", text_clean):
        cand = m.group(1).strip().rstrip(" .,:;")
        # stop at next lowercase-to-uppercase break
        cand = re.split(r"\s{2,}|(?:\s(?:the|a|an)\s)", cand)[0]
        cand = re.sub(r"\s+", " ", cand)
        if _looks_like_pharma(cand):
            found.add(cand)

    pub = company_from_url(url)
    if pub:
        found.add(pub)

    # De-noise: remove fragments that look like nouns not orgs
    clean = []
    for c in found:
        bad = re.search(r"\b(Study|Trial|Cancer|Approval|Press|Release|Medicine|FDA|Type|First|Immunotherapy)\b", c, flags=re.I)
        if not bad:
            clean.append(c)

    # Normalize casing uniqueness
    uniq = list({x.lower(): x for x in clean}.values())
    return sorted(uniq, key=lambda s: s.lower())

DRUG_SUFFIXES = (
    "mab|nib|ciclib|parib|lisib|sertib|zomib|tinib|metinib|rafenib|zumab|penib|trexate|mustine|relix"
)

def extract_drugs_all(text: str) -> List[str]:
    t = text
    drugs: Set[str] = set()

    # 1) Brand/generic pairs: e.g., atezolizumab (Tecentriq) or chemotherapy (Abraxane)
    for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9\-]{3,})\s*\(([^)]+)\)", t):
        a, b = m.group(1).strip(), m.group(2).strip()
        # Keep both if look like drug-like tokens
        if re.search(rf"({DRUG_SUFFIXES})\b", a.lower()) or len(a) >= 4:
            drugs.add(a)
        if len(b) >= 3 and not re.search(r"\s", b):  # one-token brand like Abraxane
            drugs.add(b)

    # 2) hyphenated generics like nab-paclitaxel / T-DM1
    for m in re.finditer(r"\b(?:nab-)?[A-Za-z]{3,}(?:-[A-Za-z0-9]{2,})\b", t):
        cand = m.group(0)
        if re.search(r"(paclitaxel|dm1|t-dm1|tdm1|capecitabine|cyclophosphamide|carboplatin|cisplatin)", cand, flags=re.I):
            drugs.add(cand)

    # 3) biologics/small molecules by suffix
    for m in re.finditer(rf"\b([A-Za-z][a-z][A-Za-z\-]{{2,}}(?:{DRUG_SUFFIXES}))\b", t, flags=re.I):
        drugs.add(m.group(1))

    # Clean up common non-drug words that slip in
    blacklist = {"chemotherapy"}  # not a drug name itself
    drugs = {d.strip(" ,.;:").replace("–", "-") for d in drugs if d.lower() not in blacklist}

    # Normalize case (prefer capitalized)
    def norm(d: str) -> str:
        if d.isupper():
            return d  # like T-DM1
        return d[0].upper() + d[1:] if d and d[0].isalpha() else d

    uniq = list({x.lower(): norm(x) for x in drugs}.values())
    return sorted(uniq, key=lambda s: s.lower())

def pick_indication(text: str) -> str:
    m = re.search(r"(HR\+\s*[,/]*\s*HER2-\s*.*?breast cancer|HER2\+\s*.*?breast cancer|HER2-\s*.*?breast cancer|HR\+\s*.*?breast cancer|metastatic breast cancer|early breast cancer|triple[- ]negative breast cancer|TNBC|breast cancer)", text, flags=re.I)
    if not m:
        return "breast cancer"
    ind = m.group(1)
    # Normalize TNBC expanded form
    ind = re.sub(r"\bTNBC\b", "triple-negative breast cancer", ind, flags=re.I)
    return ind

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
        tnbc=("tnbc" in lower_text or "triple-negative" in lower_text),
    )

def extract_facts(headline: str, content: str, url: str) -> Dict:
    text = f"{headline or ''}\n{content or ''}"
    text_norm = re.sub(r"\s+", " ", text).strip()
    t = text_norm.lower()

    companies = find_pharma_biotech_companies(text_norm, url)
    drugs_all = extract_drugs_all(text_norm)
    drug_primary = drugs_all[0] if drugs_all else None
    indication = pick_indication(text_norm)
    phase = pick_phase(t)
    topics = detect_topics(text_norm, t)

    return dict(
        companies=companies,
        publisher=company_from_url(url) or None,
        drug=drug_primary,
        drugs=drugs_all,
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
# LLM MINER (watsonx + HF + local Mistral)
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

    def _format_prompt(self, headline: str, url: str, content: str, facts: Dict) -> str:
        snippet = content.strip()
        if len(snippet) > 1600:
            snippet = snippet[:1600]

        facts_str = json.dumps({
            "drugs": facts["drugs"],
            "companies": facts["companies"][:5],
            "indication": facts["indication"],
            "phase": facts["phase"],
            "topics_true": [k for k, v in facts["topics"].items() if v]
        }, ensure_ascii=False)

        # STRICT relevance and uniqueness instructions
        return f"""
You are a medical content specialist creating Twitter/X poll questions about breast cancer news.

GOAL: Create several DISTINCT polls that assess PRACTICE IMPACT based ONLY on the headline and article content provided.
- Do NOT invent facts, drugs, companies, populations, lines of therapy, or endpoints that are not present.
- All proper nouns (drug names, companies, biomarkers) MUST come from the provided facts/headline/content.
- Each poll must be unique in focus (utility vs. practice change, endpoints vs. setting, biomarker vs. patient subset, etc.).

CONSTRAINTS:
- ≤ 280 characters (question + 3–4 options).
- 3–4 concise, clinically relevant options.
- Mention specific context (drug, indication, phase/setting, endpoint) when available.
- If information is missing for a dimension, omit it rather than guessing.
- Output ONLY poll blocks; no commentary.

ARTICLE INFORMATION:
Headline: "{headline}"
URL: {url}
Extracted Facts (only use these entities for proper nouns): {facts_str}

ARTICLE CONTENT (truncated):
{snippet}

FORMAT for each poll:
Q: <practice-impact question with specific context>
- Option 1
- Option 2
- Option 3
- Option 4 (optional)

Generate several DISTINCT polls now.
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

        # Try chat if present; otherwise generate_text
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

    # ----------- Anti-hallucination post-filters ----------
    @staticmethod
    def _capitalized_terms_not_in_allowed(poll: str, allowed: Set[str]) -> List[str]:
        # capture ProperCase tokens and hyphenated caps like T-DM1
        terms = set(re.findall(r"\b([A-Z][A-Za-z0-9\-]+)\b", poll))
        offenders = []
        for t in terms:
            if t in CAPS_ALLOW:
                continue
            if t.lower() in {a.lower() for a in allowed}:
                continue
            # Allow common words like Yes/No/Will/How/Is/Phase
            if re.match(r"^(Q|Yes|No|Will|How|Is|Phase|Option|New|Line|First|Second|Third)$", t):
                continue
            offenders.append(t)
        return offenders

    @staticmethod
    def _question_stem(poll: str) -> str:
        first = poll.splitlines()[0].lower()
        first = re.sub(r"[^a-z0-9 ]", " ", first)
        first = re.sub(r"\s+", " ", first).strip()
        return first

    def generate(self, headline: str, url: str, content: str, facts: Dict,
                 passes: int, temperature: float, top_p: float, max_new_tokens: int) -> List[str]:

        ready = any([self.wxa_model, getattr(self, "hf_client", None), self.gen_pipeline])
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
                    elif getattr(self, "hf_client", None):
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

        # ======== DEDUP & STRICT RELEVANCE FILTERS ========
        # 0) Basic exact dedup
        uniq = list(dict.fromkeys([normalize_ws(b) for b in all_blocks]))

        # 1) Semantic de-dup (strict)
        embedder = get_embedder()
        uniq = dedup_semantic(uniq, similarity_dup_threshold, embedder)

        # 2) Reject polls with capitalized terms not in allowed entities
        allowed_entities: Set[str] = set(facts["companies"]) | set(facts["drugs"]) | {facts["indication"]}
        if facts["phase"]:
            allowed_entities.add(f"Phase {facts['phase']}")
        allowed_entities |= {"HER2", "HR+", "HER2-", "HER2+", "TNBC", "PFS", "OS", "ORR"}

        grounded_terms = set()
        for d in facts["drugs"]:
            grounded_terms.add(d.lower())
        for c in facts["companies"]:
            grounded_terms.add(c.lower())
        for token in [facts["indication"], facts["phase"], "pik3ca", "hr+", "her2-", "her2+", "pfs", "overall survival", "os", "adjuvant", "first-line", "tnbc"]:
            if token:
                grounded_terms.add(str(token).lower())

        def grounded(poll: str) -> bool:
            t = poll.lower()
            clinical_terms_present = any(term in t for term in grounded_terms)
            practice_focus_present = any(phrase in t for phrase in [
                "practice", "clinical", "use this", "impact", "change", "inform",
                "applicable", "relevant", "approach", "treatment", "discussions"
            ])
            offenders = self._capitalized_terms_not_in_allowed(poll, allowed_entities)
            return clinical_terms_present and practice_focus_present and not offenders

        filtered = [p for p in uniq if grounded(p)]

        # 3) Intra-article uniqueness by question stem (kill near-duplicate stems)
        seen_stems = set()
        really_final = []
        for p in filtered:
            stem = self._question_stem(p)
            if all(util.cos_sim(embedder.encode(stem), embedder.encode(s)).item() < 0.92 for s in seen_stems) if seen_stems else True:
                really_final.append(p)
                seen_stems.add(stem)

        return really_final

# =========================
# RULE-BASED FALLBACK (unchanged)
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
    companies = facts["companies"]
    tp = facts["topics"]

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
    if ind and not polls:
        add(
            f"How practice-changing is this {ind} information?",
            ["Practice-changing", "Practice-informing", "Not practice-impacting", "Uncertain impact"]
        )

    embedder = get_embedder()
    polls = dedup_semantic(list(dict.fromkeys(polls)), 0.95, embedder)
    return polls

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

    if not llm_polls:
        if allow_rule_fallback:
            llm_polls = rule_based_polls(facts)
        else:
            raise RuntimeError(
                "LLM generation failed. Verify credentials/token and provider access, "
                "or enable rule-based fallback in the sidebar."
            )

    all_polls = [awareness] + llm_polls
    all_polls = list(dict.fromkeys([normalize_ws(p) for p in all_polls]))
    embedder = get_embedder()
    all_polls = dedup_semantic(all_polls, 0.95, embedder)
    if awareness in all_polls and all_polls[0] != awareness:
        all_polls.remove(awareness)
        all_polls.insert(0, awareness)
    return all_polls, facts

# =========================
# FILE INPUTS (repo root only)
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

                results.append({"headline": headline, "url": url, "polls": polls})

                with st.expander(f"#{idx}  {headline or '(no headline)'}", expanded=False):
                    if url:
                        st.markdown(f"**Source:** [{url}]({url})")
                    col1, col2, col3 = st.columns([1.2, 1, 1])
                    with col1:
                        st.markdown("**Indication:** " + (facts["indication"] or "—"))
                        st.markdown("**Drug:** " + ((facts["drug"] or "—")))
                        st.markdown("**Phase:** " + (facts["phase"] or "—"))
                    with col2:
                        st.markdown("**Publisher:** " + (facts["publisher"] or "—"))
                        st.markdown("**Companies:** " + (", ".join(facts["companies"]) or "—"))
                    with col3:
                        ts = [k for k, v in facts["topics"].items() if v]
                        st.markdown("**Topics:** " + (", ".join(ts) if ts else "—"))
                        if facts["drugs"]:
                            st.markdown("**All drugs:** " + ", ".join(facts["drugs"]))
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
    "If polls look truncated, raise 'Max new tokens per pass'. "
    "Entities are strictly extracted from the article; polls with unknown proper nouns are rejected."
)
