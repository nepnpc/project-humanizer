import re
import os
import random
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import spacy
from groq import Groq

# Load .env file automatically when running locally.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"

# ---------------------------------------------------------------------------
# System prompt — instructs the model to write like a non-native English
# speaker (Nepali-style): dropped articles, tense variation, simple vocab,
# personal asides, dramatic sentence length swings.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a Nepali university student who learned English as a second language. "
    "Your English is good but not perfect — it has a natural, non-native feel. "
    "Rewrite the given text following these rules strictly:\n"
    "1. Drop 'the', 'a', or 'an' before some nouns randomly — not always, but noticeably.\n"
    "2. Vary sentence length dramatically. Write some sentences that are only 4-6 words. "
    "Then write one long sentence that connects several ideas with 'and', 'but', or 'so'.\n"
    "3. Occasionally use present tense even when past would be correct, or vice versa.\n"
    "4. Use simple, everyday vocabulary. Avoid academic words.\n"
    "5. Add short personal observations like 'I think', 'honestly', 'from what I know', "
    "'it seems to me', 'which is interesting actually'.\n"
    "6. Sometimes start a sentence mid-thought, like 'And that is why...' or 'But still...'.\n"
    "7. Never use: 'crucial', 'delve', 'tapestry', 'furthermore', 'moreover', "
    "'it is worth noting', 'in conclusion', 'ultimately', 'commendable', 'pivotal'.\n"
    "8. Do NOT sound like a textbook or a polished essay. Sound like a real person explaining something.\n"
    "Return only the rewritten text, nothing else."
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
# Helper utilities
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    return len(text.split())


# --- Phase 1: summarise if over limit ---

def summarize_text(text: str, target_words: int) -> str:
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Condense the following text to strictly fewer than {target_words} words. "
                    "Return only the summary."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# --- Phase 2: NLP chunking (3 sentences per chunk) ---

def chunk_text(text: str, sentences_per_chunk: int = 3) -> list[str]:
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        block = " ".join(sentences[i : i + sentences_per_chunk])
        if block:
            chunks.append(block)
    return chunks


# --- Phase 3: LLM rewrite per chunk ---

def humanize_chunk(chunk: str) -> str:
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": chunk},
        ],
        temperature=0.92,
    )
    return response.choices[0].message.content.strip()


# --- Phase 4a: Contractions ---

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


# --- Phase 4b: Nepali-English grammar quirks ---

# Filler phrases injected randomly between sentences to add personal voice
_FILLERS = [
    "Honestly,", "I think", "from what I understand,",
    "it seems to me", "which is quite interesting actually,",
    "and you know,", "basically,", "I mean,",
]

# Articles that Nepali English speakers commonly drop
_ARTICLE_PATTERN = re.compile(r'\b(The|the|A|a|An|an)\s+', re.MULTILINE)


def drop_articles_randomly(text: str, drop_rate: float = 0.25) -> str:
    """Remove articles before nouns at a given probability to mimic non-native speech."""
    def maybe_drop(match: re.Match) -> str:
        return "" if random.random() < drop_rate else match.group(0)
    return _ARTICLE_PATTERN.sub(maybe_drop, text)


def inject_fillers(text: str, inject_rate: float = 0.3) -> str:
    """Inject a casual filler phrase at the start of some sentences."""
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    result = []
    for sent in sentences:
        if random.random() < inject_rate and len(sent.split()) > 6:
            filler = random.choice(_FILLERS)
            # Lowercase first char of sentence when prepending a filler
            body = sent[0].lower() + sent[1:] if sent else sent
            result.append(f"{filler} {body}")
        else:
            result.append(sent)
    return " ".join(result)


def fragment_long_sentences(text: str, threshold: int = 22) -> str:
    """
    Chop sentences that are over `threshold` words at a conjunction,
    turning one smooth sentence into two abrupt fragments — high burstiness.
    """
    doc = nlp(text)
    result = []
    for sent in doc.sents:
        words = sent.text.strip().split()
        if len(words) > threshold and random.random() < 0.5:
            # Find a mid-point conjunction to cut at
            cut = _find_conjunction_cut(words)
            if cut:
                part1 = " ".join(words[:cut]).rstrip(",")
                part2 = " ".join(words[cut:]).capitalize()
                result.append(f"{part1}. {part2}")
                continue
        result.append(sent.text.strip())
    return " ".join(result)


def _find_conjunction_cut(words: list[str]) -> int | None:
    conjunctions = {"and", "but", "so", "because", "although", "though", "while"}
    mid = len(words) // 2
    # Search around the midpoint for a conjunction
    for offset in range(0, mid):
        for i in [mid + offset, mid - offset]:
            if 3 < i < len(words) - 3 and words[i].lower() in conjunctions:
                return i
    return None


# --- Phase 4c: Homoglyph substitution ---
# Replaces a small fraction of common Latin letters with visually identical
# Unicode characters. Readers see nothing different; statistical detectors
# see a different character distribution.

_HOMOGLYPHS: dict[str, str] = {
    'a': '\u0430',  # Cyrillic а
    'e': '\u0435',  # Cyrillic е
    'o': '\u043e',  # Cyrillic о
    'p': '\u0440',  # Cyrillic р
    'c': '\u0441',  # Cyrillic с
    'x': '\u0445',  # Cyrillic х
    'i': '\u0456',  # Cyrillic Ukrainian і
}


def apply_homoglyphs(text: str, rate: float = 0.08) -> str:
    """
    Randomly swap ~8% of eligible letters with their Unicode homoglyph twins.
    Invisible to the human eye, disrupts n-gram frequency models used by detectors.
    """
    result = []
    for char in text:
        if char in _HOMOGLYPHS and random.random() < rate:
            result.append(_HOMOGLYPHS[char])
        else:
            result.append(char)
    return "".join(result)


# --- Phase 5: sentence-aware truncation ---

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

    # ── Phase 1: Ingestion & Analysis ──────────────────────────────────────
    original_count = count_words(raw_text)
    working_text = raw_text

    if original_count > target_word_limit:
        working_text = summarize_text(raw_text, target_word_limit)

    # ── Phase 2: NLP Segmentation ──────────────────────────────────────────
    chunks = chunk_text(working_text)
    if not chunks:
        return JSONResponse({"error": "Could not segment text."}, status_code=422)

    # ── Phase 3: LLM Transformation Loop ───────────────────────────────────
    humanized_chunks: list[str] = []
    for chunk in chunks:
        humanized_chunks.append(humanize_chunk(chunk))
    draft_text = " ".join(humanized_chunks)

    # ── Phase 4: Multi-layer Obfuscation ───────────────────────────────────
    draft_text = apply_contractions(draft_text)
    draft_text = strip_ai_phrases(draft_text)
    draft_text = fragment_long_sentences(draft_text)   # burstiness
    draft_text = drop_articles_randomly(draft_text)    # non-native feel
    draft_text = inject_fillers(draft_text)            # personal voice
    draft_text = apply_homoglyphs(draft_text)          # statistical noise

    # ── Phase 5: Output & Verification ─────────────────────────────────────
    draft_text = truncate_to_word_limit(draft_text, target_word_limit)
    final_count = count_words(draft_text)

    return JSONResponse({
        "original_count": original_count,
        "final_count": final_count,
        "humanized_text": draft_text,
    })
