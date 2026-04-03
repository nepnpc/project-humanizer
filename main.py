import re
import os
import math
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
FAST_MODEL = "llama-3.1-8b-instant"        # summarization only
MAIN_MODEL = "llama-3.3-70b-versatile"     # primary humanization

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

REWRITE_SYSTEM = """You rewrite text to defeat AI detectors. AI detectors measure two signals:
• PERPLEXITY — how predictable each word choice is. High perplexity = sounds human.
• BURSTINESS — how much sentence lengths vary. High burstiness = sounds human.

You must maximise both.

━━ SENTENCE LENGTH — this is mandatory, not optional ━━
• Include at least 2 sentences that are under 7 words. Short. Punchy. Declarative.
• Include at least 3 sentences between 10–20 words.
• Include no more than 1 sentence over 28 words.
• NEVER place two sentences of similar length consecutively.
• A pattern like: long → short → medium → short → long → medium is ideal.

━━ BANNED WORDS — never use these ━━
utilize, leverage, delve, crucial, vital, pivotal, comprehensive, robust, facilitate,
implement, enhance, ensure, seamlessly, streamline, innovative, cutting-edge, paradigm,
synergy, commendable, "in today's world", "in the realm of", "it is worth noting",
"plays a crucial role", "at the end of the day", "moving forward", "in conclusion",
"furthermore", "moreover", "additionally", "notably", "overall", "it goes without saying".

━━ VOICE ━━
• Use contractions everywhere they fit: it's, don't, can't, we're, they've, wouldn't, I'd.
• Start 2–3 sentences with "And", "But", or "So" — the way people actually talk.
• Add exactly one brief casual aside anywhere it feels natural: "honestly", "which is odd",
  "and that matters", "go figure", "strangely enough". Just one — don't overdo it.
• Use an em-dash — like this — once or twice for a parenthetical aside.

━━ OUTPUT RULES ━━
• Return ONLY the rewritten text. No preamble, no "Here is the rewritten text:", nothing.
• Do not add bullet points, headers, or numbered lists.
• Do not change facts or meaning — only change how it's said."""

POLISH_SYSTEM = """Do a tight final edit on this text. Make these specific fixes only:

1. RHYTHM: Find any two consecutive sentences within 4 words of each other in length.
   Either split the longer one into two, or merge one of them with a neighbor.
2. TRANSITIONS: Remove any AI-sounding transition at the start of a sentence:
   "In conclusion", "Furthermore", "Moreover", "Additionally", "Notably", "Overall",
   "It is worth noting that", "It goes without saying".
3. STIFFNESS: Replace any formal or stiff word with a plain spoken alternative.
4. CONTRACTIONS: Apply contractions wherever they'd sound natural.

Return ONLY the edited text. No explanation."""

# ─────────────────────────────────────────────────────────────────────────────
# App lifespan
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing tables
# ─────────────────────────────────────────────────────────────────────────────

# AI cliché replacement — catches whatever the LLM still slips through
_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\butilize\b", re.IGNORECASE), "use"),
    (re.compile(r"\butilization\b", re.IGNORECASE), "use"),
    (re.compile(r"\bleverage\b", re.IGNORECASE), "use"),
    (re.compile(r"\bdelve\b", re.IGNORECASE), "dig"),
    (re.compile(r"\bfacilitate\b", re.IGNORECASE), "help"),
    (re.compile(r"\bcommence\b", re.IGNORECASE), "start"),
    (re.compile(r"\benhance\b", re.IGNORECASE), "improve"),
    (re.compile(r"\bseamlessly\b", re.IGNORECASE), "smoothly"),
    (re.compile(r"\bcutting-edge\b", re.IGNORECASE), "latest"),
    (re.compile(r"\binnovative\b", re.IGNORECASE), "new"),
    (re.compile(r"\brobust\b", re.IGNORECASE), "solid"),
    (re.compile(r"\bcomprehensive\b", re.IGNORECASE), "thorough"),
    (re.compile(r"\bpivotal\b", re.IGNORECASE), "key"),
    (re.compile(r"\bcrucial\b", re.IGNORECASE), "important"),
    (re.compile(r"\bgame-changer\b", re.IGNORECASE), "big shift"),
    (re.compile(r"\bparadigm shift\b", re.IGNORECASE), "major change"),
    (re.compile(r"\bsynergy\b", re.IGNORECASE), "teamwork"),
    (re.compile(r"\bit is worth noting that\s*", re.IGNORECASE), ""),
    (re.compile(r"\bit is important to note that\s*", re.IGNORECASE), ""),
    (re.compile(r"\bin today's (?:fast-paced )?world\b", re.IGNORECASE), "these days"),
    (re.compile(r"\bin the realm of\b", re.IGNORECASE), "in"),
    (re.compile(r"\bgame-changer\b", re.IGNORECASE), "big deal"),
    (re.compile(r"\bin conclusion,?\s*", re.IGNORECASE), ""),
    (re.compile(r"\bfurthermore,?\s*", re.IGNORECASE), ""),
    (re.compile(r"\bmoreover,?\s*", re.IGNORECASE), ""),
    (re.compile(r"\badditionally,?\s*", re.IGNORECASE), ""),
    (re.compile(r"\bnotably,?\s*", re.IGNORECASE), ""),
    (re.compile(r"\boverall,?\s*", re.IGNORECASE), ""),
    (re.compile(r"\bit goes without saying\s*(?:that)?\s*", re.IGNORECASE), ""),
]

_CONTRACTIONS: list[tuple[re.Pattern, str]] = [
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


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────

def count_words(text: str) -> int:
    return len(text.split())


def get_sentences(text: str) -> list[str]:
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def burstiness_score(sentences: list[str]) -> float:
    """
    Coefficient of variation of sentence word-lengths.
    Human text: typically 0.55–0.90+
    AI text: typically 0.25–0.45
    """
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) < 2:
        return 1.0
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.0
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    return math.sqrt(variance) / mean


def apply_replacements(text: str) -> str:
    for pattern, replacement in _REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = re.sub(r" {2,}", " ", text).strip()
    # Fix capitalisation after an empty-replacement wipe at sentence start
    text = re.sub(r"\.\s+([a-z])", lambda m: ". " + m.group(1).upper(), text)
    return text


def apply_contractions(text: str) -> str:
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    return text


def enforce_burstiness(text: str, target_cv: float = 0.55) -> str:
    """
    Measure sentence-length variance. If below target_cv,
    split the longest sentences at a natural conjunction break
    to inject short-sentence contrast.
    """
    sentences = get_sentences(text)
    if burstiness_score(sentences) >= target_cv:
        return text  # already bursty enough

    conjunctions = {"and", "but", "so", "because", "although", "though", "while", "since", "yet"}
    modified: dict[int, str] = dict(enumerate(sentences))
    max_splits = max(1, len(sentences) // 3)
    split_count = 0

    # Process from longest sentence downward
    by_length = sorted(modified.keys(), key=lambda i: len(modified[i].split()), reverse=True)

    for idx in by_length:
        if split_count >= max_splits:
            break
        words = modified[idx].split()
        if len(words) < 20:
            break
        mid = len(words) // 2
        cut: int | None = None
        for offset in range(0, mid - 3):
            for i in [mid + offset, mid - offset]:
                if 4 < i < len(words) - 4 and words[i].lower() in conjunctions:
                    cut = i
                    break
            if cut is not None:
                break
        if cut is not None:
            part1 = " ".join(words[:cut]).rstrip(",") + "."
            part2 = words[cut][0].upper() + words[cut][1:] + " " + " ".join(words[cut + 1:])
            modified[idx] = f"{part1} {part2}"
            split_count += 1

    return " ".join(modified[i] for i in sorted(modified))


def chunk_text(text: str, sentences_per_chunk: int = 5) -> list[str]:
    """
    Larger chunks (5 sentences) keep more context for the LLM,
    producing better rhythm and coherence than 3-sentence chunks.
    """
    sentences = get_sentences(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        block = " ".join(sentences[i: i + sentences_per_chunk])
        if block:
            chunks.append(block)
    return chunks


def truncate_to_word_limit(text: str, limit: int) -> str:
    if count_words(text) <= limit:
        return text
    sentences = get_sentences(text)
    result: list[str] = []
    total = 0
    for sent in sentences:
        sw = len(sent.split())
        if total + sw <= limit:
            result.append(sent)
            total += sw
        else:
            break
    return " ".join(result)


# ─────────────────────────────────────────────────────────────────────────────
# LLM calls
# ─────────────────────────────────────────────────────────────────────────────

def llm(system: str, user_text: str, model: str, temperature: float) -> str:
    response = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def summarize_text(text: str, target_words: int) -> str:
    return llm(
        system=(
            f"Condense the following text to strictly fewer than {target_words} words. "
            "Preserve all key ideas. Return only the condensed text."
        ),
        user_text=text,
        model=FAST_MODEL,
        temperature=0.3,
    )


def humanize_chunk(chunk: str) -> str:
    return llm(
        system=REWRITE_SYSTEM,
        user_text=chunk,
        model=MAIN_MODEL,
        temperature=0.92,
    )


def polish_pass(text: str) -> str:
    return llm(
        system=POLISH_SYSTEM,
        user_text=text,
        model=MAIN_MODEL,
        temperature=0.72,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

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

    original_count = count_words(raw_text)
    working_text = raw_text

    # Phase 1: Summarise if significantly over limit
    if original_count > target_word_limit * 1.25:
        working_text = summarize_text(raw_text, target_word_limit)

    # Phase 2: Chunk into 5-sentence groups for coherence
    chunks = chunk_text(working_text, sentences_per_chunk=5)
    if not chunks:
        return JSONResponse({"error": "Could not segment text."}, status_code=422)

    # Phase 3: Primary LLM rewrite (llama-3.3-70b-versatile per chunk)
    humanized_chunks = [humanize_chunk(c) for c in chunks]
    draft = " ".join(humanized_chunks)

    # Phase 4: Post-processing — clichés, contractions, burstiness
    draft = apply_replacements(draft)
    draft = apply_contractions(draft)
    draft = enforce_burstiness(draft)

    # Phase 5: Measure burstiness; run polish pass only if still low
    sentences = get_sentences(draft)
    score = burstiness_score(sentences)
    ran_polish = False
    if score < 0.45:
        draft = polish_pass(draft)
        draft = apply_replacements(draft)
        draft = apply_contractions(draft)
        sentences = get_sentences(draft)
        score = burstiness_score(sentences)
        ran_polish = True

    # Phase 6: Truncate to word limit
    draft = truncate_to_word_limit(draft, target_word_limit)
    final_count = count_words(draft)

    return JSONResponse({
        "original_count": original_count,
        "final_count": final_count,
        "humanized_text": draft,
        "burstiness_score": round(score, 3),
        "polish_applied": ran_polish,
    })
