import re
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import spacy
from groq import Groq

# Load .env file automatically when running locally (ignored in production).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — rely on real environment variables

# ---------------------------------------------------------------------------
# Configuration — set GROQ_API_KEY as an environment variable.
# Locally: create a .env file with GROQ_API_KEY=your_key
# Production: set it in your host's environment/secrets dashboard.
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama3-8b-8192"

SYSTEM_PROMPT = (
    "You are an expert human writer. Rewrite this text snippet. "
    "Drastically vary sentence length (burstiness). "
    "Combine some sentences using conjunctions; chop others into short statements. "
    "Use vivid, slightly unconventional vocabulary. "
    "Never use transition words like 'crucial', 'delve', 'tapestry', 'furthermore', or 'moreover'. "
    "Maintain an empathetic tone. Return only the rewritten text."
)

# ---------------------------------------------------------------------------
# App lifespan — load the spaCy model once at startup
# ---------------------------------------------------------------------------
nlp: spacy.Language | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global nlp
    nlp = spacy.load("en_core_web_sm")
    yield


app = FastAPI(title="Project Humanizer", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
client = Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    return len(text.split())


# --- Phase 1 helper: summarise if over limit ---

def summarize_text(text: str, target_words: int) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a precise summarizer. Condense the following text to strictly "
                    f"fewer than {target_words} words. Return only the summary, nothing else."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# --- Phase 2 helper: NLP chunking (3–4 sentences per chunk) ---

def chunk_text(text: str, sentences_per_chunk: int = 3) -> list[str]:
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        block = " ".join(sentences[i : i + sentences_per_chunk])
        if block:
            chunks.append(block)
    return chunks


# --- Phase 3 helper: single-chunk LLM rewrite ---

def humanize_chunk(chunk: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": chunk},
        ],
        temperature=0.85,
    )
    return response.choices[0].message.content.strip()


# --- Phase 4 helper: regex post-processing ---

# Map of (compiled pattern, replacement) pairs.
# Order matters — longer phrases must precede their sub-phrases.
_CONTRACTION_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bcannot\b", re.IGNORECASE), "can't"),
    (re.compile(r"\bwill not\b", re.IGNORECASE), "won't"),
    (re.compile(r"\bwould not\b", re.IGNORECASE), "wouldn't"),
    (re.compile(r"\bcould not\b", re.IGNORECASE), "couldn't"),
    (re.compile(r"\bshould not\b", re.IGNORECASE), "shouldn't"),
    (re.compile(r"\bdoes not\b", re.IGNORECASE), "doesn't"),
    (re.compile(r"\bdid not\b", re.IGNORECASE), "didn't"),
    (re.compile(r"\bdo not\b", re.IGNORECASE), "don't"),
    (re.compile(r"\bare not\b", re.IGNORECASE), "aren't"),
    (re.compile(r"\bwere not\b", re.IGNORECASE), "weren't"),
    (re.compile(r"\bwas not\b", re.IGNORECASE), "wasn't"),
    (re.compile(r"\bis not\b", re.IGNORECASE), "isn't"),
    (re.compile(r"\bhave not\b", re.IGNORECASE), "haven't"),
    (re.compile(r"\bhas not\b", re.IGNORECASE), "hasn't"),
    (re.compile(r"\bhad not\b", re.IGNORECASE), "hadn't"),
    (re.compile(r"\bI am\b"), "I'm"),
    (re.compile(r"\bI have\b"), "I've"),
    (re.compile(r"\bI will\b"), "I'll"),
    (re.compile(r"\bI would\b"), "I'd"),
    (re.compile(r"\bthey are\b", re.IGNORECASE), "they're"),
    (re.compile(r"\bthey have\b", re.IGNORECASE), "they've"),
    (re.compile(r"\bthey will\b", re.IGNORECASE), "they'll"),
    (re.compile(r"\bwe are\b", re.IGNORECASE), "we're"),
    (re.compile(r"\bwe have\b", re.IGNORECASE), "we've"),
    (re.compile(r"\bwe will\b", re.IGNORECASE), "we'll"),
    (re.compile(r"\byou are\b", re.IGNORECASE), "you're"),
    (re.compile(r"\byou have\b", re.IGNORECASE), "you've"),
    (re.compile(r"\byou will\b", re.IGNORECASE), "you'll"),
    (re.compile(r"\bhe is\b", re.IGNORECASE), "he's"),
    (re.compile(r"\bshe is\b", re.IGNORECASE), "she's"),
    (re.compile(r"\bit is\b", re.IGNORECASE), "it's"),
    (re.compile(r"\bthat is\b", re.IGNORECASE), "that's"),
    (re.compile(r"\bthere is\b", re.IGNORECASE), "there's"),
    (re.compile(r"\bwhat is\b", re.IGNORECASE), "what's"),
    (re.compile(r"\bwho is\b", re.IGNORECASE), "who's"),
]

# Phrases stripped from the start of any sentence / paragraph
_AI_OPENER_PATTERN = re.compile(
    r"^(?:"
    r"In conclusion,?\s*|"
    r"Ultimately,?\s*|"
    r"To summarize,?\s*|"
    r"In summary,?\s*|"
    r"To conclude,?\s*|"
    r"In closing,?\s*|"
    r"As a result,?\s*|"
    r"Therefore,?\s*|"
    r"Thus,?\s*|"
    r"Hence,?\s*|"
    r"It(?:'s| is) worth noting that\s*|"
    r"It is important to note that\s*|"
    r"Notably,?\s*|"
    r"Overall,?\s*|"
    r"All in all,?\s*"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def apply_contractions(text: str) -> str:
    for pattern, replacement in _CONTRACTION_RULES:
        text = pattern.sub(replacement, text)
    return text


def strip_ai_phrases(text: str) -> str:
    return _AI_OPENER_PATTERN.sub("", text).strip()


# --- Phase 5 helper: sentence-aware truncation ---

def truncate_to_word_limit(text: str, limit: int) -> str:
    if count_words(text) <= limit:
        return text

    doc = nlp(text)
    result: list[str] = []
    word_count = 0
    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_words = len(sent_text.split())
        if word_count + sent_words <= limit:
            result.append(sent_text)
            word_count += sent_words
        else:
            break
    return " ".join(result)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/humanize")
async def humanize(request: Request):
    data = await request.json()
    raw_text: str = data.get("raw_text", "").strip()
    target_word_limit: int = int(data.get("target_word_limit", 500))

    if not raw_text:
        return JSONResponse({"error": "No text provided."}, status_code=400)

    if target_word_limit < 10:
        return JSONResponse({"error": "Target word limit must be at least 10."}, status_code=400)

    # ── Phase 1: Ingestion & Analysis ──────────────────────────────────────
    original_count = count_words(raw_text)
    working_text = raw_text

    if original_count > target_word_limit:
        working_text = summarize_text(raw_text, target_word_limit)

    # ── Phase 2: NLP Segmentation ──────────────────────────────────────────
    chunks = chunk_text(working_text)

    if not chunks:
        return JSONResponse({"error": "Could not segment the provided text."}, status_code=422)

    # ── Phase 3: LLM Transformation Loop ───────────────────────────────────
    humanized_chunks: list[str] = []
    for chunk in chunks:
        rewritten = humanize_chunk(chunk)
        humanized_chunks.append(rewritten)

    draft_text = " ".join(humanized_chunks)

    # ── Phase 4: Mechanical Obfuscation ────────────────────────────────────
    draft_text = apply_contractions(draft_text)
    draft_text = strip_ai_phrases(draft_text)

    # ── Phase 5: Output & Verification ─────────────────────────────────────
    draft_text = truncate_to_word_limit(draft_text, target_word_limit)
    final_count = count_words(draft_text)

    return JSONResponse(
        {
            "original_count": original_count,
            "final_count": final_count,
            "humanized_text": draft_text,
        }
    )
