import re
import os
import random
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import spacy
from groq import Groq

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"

# ---------------------------------------------------------------------------
# System prompt — target is readable, natural, conversational human writing.
# High burstiness and perplexity come from rhythm and word choice,
# NOT from grammatical errors or broken English.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Rewrite the following text so it sounds like a real person wrote it — "
    "conversational, thoughtful, and a little informal. Follow these rules:\n\n"
    "1. SENTENCE LENGTH: Mix very short sentences (3–6 words) with longer flowing ones. "
    "Never write three sentences of the same length in a row.\n"
    "2. RHYTHM: Start some sentences with 'And', 'But', or 'So' for flow. "
    "Use em-dashes — like this — to insert asides.\n"
    "3. VOCABULARY: Prefer vivid, concrete words over generic ones. "
    "Avoid: 'crucial', 'delve', 'tapestry', 'foster', 'pivotal', 'commendable', "
    "'it is worth noting', 'in conclusion', 'moreover', 'furthermore', 'ultimately'.\n"
    "4. PERSONAL VOICE: Add one or two short personal observations like "
    "'honestly', 'which is strange', 'and that matters', 'go figure'. "
    "Keep them brief — don't overdo it.\n"
    "5. CONTRACTIONS: Use them naturally — it's, don't, they're, wasn't.\n"
    "6. DO NOT: lecture, summarise, or write like an essay. "
    "Sound like someone telling you something interesting, not writing a report.\n\n"
    "Return only the rewritten text. No intro, no commentary."
)

# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------
nlp: spacy.Language | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global nlp
    nlp = spacy.load("en_core_web_sm")
    yield


app = FastAPI(title="Project Humanizer", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY)
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        _client = Groq(api_key=key)
    return _client


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": f"{type(exc).__name__}: {str(exc)}"},
    )


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    return len(text.split())


# Phase 1: optional summarisation

def summarize_text(text: str, target_words: int) -> str:
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Condense the following text to strictly fewer than {target_words} words. "
                    "Preserve the key ideas. Return only the condensed text."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# Phase 2: NLP chunking

def chunk_text(text: str, sentences_per_chunk: int = 3) -> list[str]:
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        block = " ".join(sentences[i : i + sentences_per_chunk])
        if block:
            chunks.append(block)
    return chunks


# Phase 3: LLM rewrite

def humanize_chunk(chunk: str) -> str:
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": chunk},
        ],
        temperature=0.88,
    )
    return response.choices[0].message.content.strip()


# Phase 4a: contractions

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


# Phase 4b: subtle burstiness enforcement
# Splits overly long sentences at a natural conjunction point.
# This is the ONE mechanical trick that genuinely raises burstiness score
# without making the text look wrong.

def _find_conjunction_cut(words: list[str]) -> int | None:
    conjunctions = {"and", "but", "so", "because", "although", "though", "while", "since"}
    mid = len(words) // 2
    for offset in range(0, mid - 2):
        for i in [mid + offset, mid - offset]:
            if 4 < i < len(words) - 4 and words[i].lower() in conjunctions:
                return i
    return None


def enforce_burstiness(text: str, long_threshold: int = 28) -> str:
    """
    Split sentences over `long_threshold` words at a mid-point conjunction.
    Only fires on ~60% of eligible sentences to avoid over-processing.
    """
    doc = nlp(text)
    result = []
    for sent in doc.sents:
        words = sent.text.strip().split()
        if len(words) > long_threshold and random.random() < 0.6:
            cut = _find_conjunction_cut(words)
            if cut:
                part1 = " ".join(words[:cut]).rstrip(",")
                part2 = " ".join(words[cut:])
                # Capitalise start of part2 if it begins with a conjunction
                part2 = part2[0].upper() + part2[1:]
                result.append(f"{part1}. {part2}")
                continue
        result.append(sent.text.strip())
    return " ".join(result)


# Phase 5: sentence-aware truncation

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

@app.get("/health")
async def health():
    key_set = bool(os.environ.get("GROQ_API_KEY", GROQ_API_KEY))
    return {"status": "ok", "groq_key_set": key_set, "spacy_loaded": nlp is not None}


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

    # Phase 1: Summarise if over limit
    original_count = count_words(raw_text)
    working_text = raw_text
    if original_count > target_word_limit:
        working_text = summarize_text(raw_text, target_word_limit)

    # Phase 2: Chunk
    chunks = chunk_text(working_text)
    if not chunks:
        return JSONResponse({"error": "Could not segment text."}, status_code=422)

    # Phase 3: LLM rewrite
    humanized_chunks = [humanize_chunk(c) for c in chunks]
    draft_text = " ".join(humanized_chunks)

    # Phase 4: Post-processing
    draft_text = apply_contractions(draft_text)
    draft_text = strip_ai_phrases(draft_text)
    draft_text = enforce_burstiness(draft_text)

    # Phase 5: Truncate & return
    draft_text = truncate_to_word_limit(draft_text, target_word_limit)
    final_count = count_words(draft_text)

    return JSONResponse({
        "original_count": original_count,
        "final_count": final_count,
        "humanized_text": draft_text,
    })
