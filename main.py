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
FAST_MODEL = "llama-3.1-8b-instant"
MAIN_MODEL = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────────────────────
# Prompts — academic report style
# ─────────────────────────────────────────────────────────────────────────────

ACADEMIC_SYSTEM = """You are a diligent third-year university student rewriting a paragraph from your formal report submission. Your writing is competent and clear, but it is student writing — not a polished journal article and not AI-generated text.

━━ REGISTER ━━
• Formal throughout. No slang, no casual language, no contractions at all.
  Write "it is", "do not", "cannot", "they are" — never "it's", "don't", "can't", "they're".
• Third person unless the original explicitly uses first person.
• Clear and direct — not ornate. Students do not over-write.

━━ SENTENCE STRUCTURE — students mix these naturally ━━
• Short declarative: "This indicates a significant difference."
  "The results were consistent with the hypothesis."
• Medium analytical: "This can be attributed to the fact that X, which in turn affects Y."
• Longer with subordination: "Although X tends to Y under certain conditions,
  the data collected in this study suggest that Z may also be a contributing factor."
• Vary length across the paragraph. Avoid three sentences of the same length in a row.
  Academic writing is not uniform — short summary sentences sit beside longer explanations.

━━ VOICE — the most important marker of student writing ━━
• PASSIVE for findings, observations, and methods:
  "it was found that", "it was observed that", "it can be seen that",
  "the results indicate", "this was attributed to", "it has been noted that",
  "it was determined that", "data were collected", "measurements were taken".
• ACTIVE for what the report itself does:
  "This section examines", "This report investigates", "The analysis considers",
  "This study aims to", "The following section discusses".

━━ HEDGING — students rarely make absolute claims ━━
Use appropriately: "it can be argued that", "the evidence suggests", "this appears to indicate",
"it is possible that", "to some extent", "this may be due to", "it seems likely that",
"the data suggest" (not "the data prove").

━━ TRANSITIONS STUDENTS ACTUALLY USE ━━
Openers: "Furthermore,", "In addition,", "However,", "Nevertheless,", "As a result,",
"In contrast,", "Similarly,", "With regard to", "In terms of", "In relation to",
"Building on this,", "This is further supported by"
Back-references: "As discussed above,", "As noted previously,", "In line with this,",
"Consistent with these findings,", "As shown in the previous section,"
Conclusions of a paragraph: "This suggests that", "This indicates that",
"This supports the view that", "This can be explained by"

━━ VOCABULARY ━━
• Repeat key technical terms — students do not constantly rephrase subject-specific words.
• Prefer clear over impressive. "use" is fine. "examine" is fine. "understand" is fine.
• Academic alternatives to common AI words:
  utilize → use | leverage → apply | delve → examine | facilitate → support | enable
  innovative → novel | cutting-edge → recent | robust → reliable | comprehensive → thorough
  seamlessly → effectively | enhance → improve | commence → begin
• NEVER use: "it is worth noting that", "it is important to note that",
  "it goes without saying", "in today's world", "in the realm of", "at the end of the day",
  "moving forward", "game-changer", "paradigm shift", "synergy".

━━ PARAGRAPH STRUCTURE ━━
• Topic sentence states the point of the paragraph.
• 2–4 sentences develop, explain, or provide evidence.
• Final sentence often links forward or draws a conclusion from the paragraph's evidence.
• One clear idea per paragraph — do not mix unrelated points.

━━ OUTPUT ━━
• Return ONLY the rewritten text.
• Preserve any headers, bullet points, or numbered lists from the original exactly.
• Do not restructure, reorder, or add new sections.
• No preamble ("Here is the rewritten version…"), no labels, no commentary."""

ACADEMIC_POLISH = """You are doing a final check on a paragraph from a university student's report. Make only these corrections:

1. CONTRACTIONS: Expand every contraction — "it's" → "it is", "don't" → "do not",
   "can't" → "cannot", "won't" → "will not", "they're" → "they are", and so on.
   Academic reports do not use contractions.

2. AI PHRASES: Remove or replace these specific phrases:
   "it is worth noting that" → remove or rephrase as "Of note,"
   "it is important to note that" → remove or rephrase naturally
   "it goes without saying" → remove entirely
   "in today's world" → "in recent years" or "currently"
   "in conclusion," → "To summarise," or "In summary,"
   "at the end of the day" → remove or rephrase
   "moving forward" → rephrase naturally

3. PASSIVE/ACTIVE: Findings and observations should use passive voice.
   If a finding sentence is in active voice with "AI-style" construction, convert it.
   Example: "The data clearly shows that X is crucial" →
            "The data suggest that X plays an important role"

4. HEDGING: Any absolute statement about uncertain findings should be hedged:
   "proves" → "suggests" | "shows that" → "indicates that" | "clearly" → consider removing

Return ONLY the corrected text. No explanation, no labels."""

# ─────────────────────────────────────────────────────────────────────────────
# App lifespan
# ─────────────────────────────────────────────────────────────────────────────

nlp: spacy.Language | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global nlp
    nlp = spacy.load("en_core_web_sm")
    yield


app = FastAPI(title="Project Humanizer — Academic", lifespan=lifespan)
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

# Expand contractions — academic reports do not use them
_EXPAND_CONTRACTIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bcan't\b", re.IGNORECASE), "cannot"),
    (re.compile(r"\bwon't\b", re.IGNORECASE), "will not"),
    (re.compile(r"\bwouldn't\b", re.IGNORECASE), "would not"),
    (re.compile(r"\bcouldn't\b", re.IGNORECASE), "could not"),
    (re.compile(r"\bshouldn't\b", re.IGNORECASE), "should not"),
    (re.compile(r"\bdoesn't\b", re.IGNORECASE), "does not"),
    (re.compile(r"\bdidn't\b", re.IGNORECASE), "did not"),
    (re.compile(r"\bdon't\b", re.IGNORECASE), "do not"),
    (re.compile(r"\baren't\b", re.IGNORECASE), "are not"),
    (re.compile(r"\bweren't\b", re.IGNORECASE), "were not"),
    (re.compile(r"\bwasn't\b", re.IGNORECASE), "was not"),
    (re.compile(r"\bisn't\b", re.IGNORECASE), "is not"),
    (re.compile(r"\bhaven't\b", re.IGNORECASE), "have not"),
    (re.compile(r"\bhasn't\b", re.IGNORECASE), "has not"),
    (re.compile(r"\bhadn't\b", re.IGNORECASE), "had not"),
    (re.compile(r"\bI'm\b"), "I am"),
    (re.compile(r"\bI've\b"), "I have"),
    (re.compile(r"\bI'll\b"), "I will"),
    (re.compile(r"\bI'd\b"), "I would"),
    (re.compile(r"\bthey're\b", re.IGNORECASE), "they are"),
    (re.compile(r"\bthey've\b", re.IGNORECASE), "they have"),
    (re.compile(r"\bthey'll\b", re.IGNORECASE), "they will"),
    (re.compile(r"\bwe're\b", re.IGNORECASE), "we are"),
    (re.compile(r"\bwe've\b", re.IGNORECASE), "we have"),
    (re.compile(r"\bwe'll\b", re.IGNORECASE), "we will"),
    (re.compile(r"\byou're\b", re.IGNORECASE), "you are"),
    (re.compile(r"\byou've\b", re.IGNORECASE), "you have"),
    (re.compile(r"\byou'll\b", re.IGNORECASE), "you will"),
    (re.compile(r"\bhe's\b", re.IGNORECASE), "he is"),
    (re.compile(r"\bshe's\b", re.IGNORECASE), "she is"),
    (re.compile(r"\bit's\b", re.IGNORECASE), "it is"),
    (re.compile(r"\bthat's\b", re.IGNORECASE), "that is"),
    (re.compile(r"\bthere's\b", re.IGNORECASE), "there is"),
    (re.compile(r"\bwhat's\b", re.IGNORECASE), "what is"),
    (re.compile(r"\bwho's\b", re.IGNORECASE), "who is"),
    (re.compile(r"\bwhere's\b", re.IGNORECASE), "where is"),
    (re.compile(r"\bthey'd\b", re.IGNORECASE), "they would"),
    (re.compile(r"\bwe'd\b", re.IGNORECASE), "we would"),
    (re.compile(r"\bhe'd\b", re.IGNORECASE), "he would"),
    (re.compile(r"\bshe'd\b", re.IGNORECASE), "she would"),
]

# AI cliché removal — catches whatever slips through the LLM rewrite
_AI_PHRASES: list[tuple[re.Pattern, str]] = [
    # Hard removes (phrase is just filler)
    (re.compile(r"\bit is worth noting that\s*", re.IGNORECASE), ""),
    (re.compile(r"\bit is important to note that\s*", re.IGNORECASE), ""),
    (re.compile(r"\bit is important to\s+(?:understand|recognise|recognize)\s+that\s*", re.IGNORECASE), ""),
    (re.compile(r"\bit goes without saying\s*(?:that)?\s*", re.IGNORECASE), ""),
    (re.compile(r"\bat the end of the day\b", re.IGNORECASE), "ultimately"),
    (re.compile(r"\bmoving forward\b", re.IGNORECASE), "going forward"),
    # Academic-appropriate replacements
    (re.compile(r"\bin today's (?:fast-paced )?world\b", re.IGNORECASE), "in recent years"),
    (re.compile(r"\bin the realm of\b", re.IGNORECASE), "in the field of"),
    (re.compile(r"\bin conclusion,?\s*", re.IGNORECASE), "To summarise, "),
    (re.compile(r"\bin summary,?\s*", re.IGNORECASE), "To summarise, "),
    # Word-level replacements — academic versions
    (re.compile(r"\butilize\b", re.IGNORECASE), "use"),
    (re.compile(r"\butilization\b", re.IGNORECASE), "use"),
    (re.compile(r"\bleverage\b", re.IGNORECASE), "apply"),
    (re.compile(r"\bdelve\b", re.IGNORECASE), "examine"),
    (re.compile(r"\bfacilitate\b", re.IGNORECASE), "support"),
    (re.compile(r"\bcommence\b", re.IGNORECASE), "begin"),
    (re.compile(r"\bseamlessly\b", re.IGNORECASE), "effectively"),
    (re.compile(r"\bcutting-edge\b", re.IGNORECASE), "recent"),
    (re.compile(r"\binnovative\b", re.IGNORECASE), "novel"),
    (re.compile(r"\brobust\b", re.IGNORECASE), "reliable"),
    (re.compile(r"\bcomprehensive\b", re.IGNORECASE), "thorough"),
    (re.compile(r"\bpivotal\b", re.IGNORECASE), "significant"),
    (re.compile(r"\bcrucial\b", re.IGNORECASE), "important"),
    (re.compile(r"\bgame-changer\b", re.IGNORECASE), "major development"),
    (re.compile(r"\bparadigm shift\b", re.IGNORECASE), "significant change"),
    (re.compile(r"\bsynergy\b", re.IGNORECASE), "collaboration"),
    (re.compile(r"\benhance\b", re.IGNORECASE), "improve"),
    (re.compile(r"\bensure\b", re.IGNORECASE), "ensure"),  # "ensure" is fine in academic
    (re.compile(r"\bstreamline\b", re.IGNORECASE), "simplify"),
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
    Academic student writing: typically 0.35–0.60
    AI academic text: typically 0.15–0.32 (very uniform)
    """
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) < 2:
        return 1.0
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.0
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    return math.sqrt(variance) / mean


def is_structural_line(text: str) -> bool:
    """
    Returns True for lines that look like headers, section titles, or
    list labels — things that should be passed through unchanged.
    """
    stripped = text.strip()
    if not stripped:
        return False
    word_count = len(stripped.split())
    # Short, no terminal sentence punctuation, possibly numbered
    no_terminal = not stripped[-1] in ".!?"
    return word_count <= 7 and no_terminal


def split_into_paragraphs(text: str) -> list[str]:
    """Split on blank lines, preserving the paragraph structure of a report."""
    paragraphs = re.split(r"\n[ \t]*\n", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def expand_contractions(text: str) -> str:
    """Expand all contractions — academic reports do not use them."""
    for pattern, replacement in _EXPAND_CONTRACTIONS:
        text = pattern.sub(replacement, text)
    return text


def apply_ai_phrase_cleanup(text: str) -> str:
    """Remove / replace AI clichés that slip past the LLM rewrite."""
    for pattern, replacement in _AI_PHRASES:
        text = pattern.sub(replacement, text)
    # Clean up double spaces from phrase removals
    text = re.sub(r" {2,}", " ", text).strip()
    # Restore capitalisation after a phrase was wiped at sentence start
    text = re.sub(r"\.\s+([a-z])", lambda m: ". " + m.group(1).upper(), text)
    return text


def enforce_sentence_variety(text: str, min_cv: float = 0.30) -> str:
    """
    For academic text the target CV is lower than casual writing (~0.35–0.50).
    Only intervene if sentences are extremely uniform (CV < min_cv).
    Splits the longest sentences at a subordinating conjunction.
    """
    sentences = get_sentences(text)
    if burstiness_score(sentences) >= min_cv:
        return text

    # Conjunctions acceptable to break on in academic writing
    break_words = {"although", "though", "while", "whereas", "because",
                   "since", "however", "which", "and", "but"}

    modified: dict[int, str] = dict(enumerate(sentences))
    max_splits = max(1, len(sentences) // 4)
    split_count = 0

    by_length = sorted(modified.keys(), key=lambda i: len(modified[i].split()), reverse=True)

    for idx in by_length:
        if split_count >= max_splits:
            break
        words = modified[idx].split()
        if len(words) < 22:
            break
        mid = len(words) // 2
        cut: int | None = None
        for offset in range(0, mid - 4):
            for i in [mid + offset, mid - offset]:
                if 5 < i < len(words) - 5 and words[i].lower() in break_words:
                    cut = i
                    break
            if cut is not None:
                break
        if cut is not None:
            part1 = " ".join(words[:cut]).rstrip(",") + "."
            w = words[cut]
            part2 = w[0].upper() + w[1:] + " " + " ".join(words[cut + 1:])
            modified[idx] = f"{part1} {part2}"
            split_count += 1

    return " ".join(modified[i] for i in sorted(modified))


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
            "Preserve all key arguments, findings, and structure. "
            "Maintain formal academic language. Return only the condensed text."
        ),
        user_text=text,
        model=FAST_MODEL,
        temperature=0.25,
    )


def rewrite_paragraph(para: str) -> str:
    """Rewrite a single paragraph in student academic style."""
    return llm(
        system=ACADEMIC_SYSTEM,
        user_text=para,
        model=MAIN_MODEL,
        temperature=0.85,
    )


def polish_paragraph(para: str) -> str:
    """Final check on a paragraph — contractions, AI phrases, passive voice."""
    return llm(
        system=ACADEMIC_POLISH,
        user_text=para,
        model=MAIN_MODEL,
        temperature=0.60,
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

    # Phase 2: Split by paragraph — preserves report structure
    paragraphs = split_into_paragraphs(working_text)
    if not paragraphs:
        return JSONResponse({"error": "Could not segment text."}, status_code=422)

    # Phase 3: Rewrite each paragraph in academic student style
    rewritten: list[str] = []
    for para in paragraphs:
        if is_structural_line(para):
            rewritten.append(para)  # pass headers through unchanged
        elif count_words(para) < 8:
            rewritten.append(para)  # too short to rewrite meaningfully
        else:
            rewritten.append(rewrite_paragraph(para))

    draft = "\n\n".join(rewritten)

    # Phase 4: Post-processing — expand contractions, remove AI clichés
    draft = expand_contractions(draft)
    draft = apply_ai_phrase_cleanup(draft)

    # Phase 5: Measure sentence variety per paragraph;
    #           run polish pass on paragraphs that are still too uniform
    processed_paragraphs = split_into_paragraphs(draft)
    polished: list[str] = []
    ran_polish = False

    for para in processed_paragraphs:
        if is_structural_line(para) or count_words(para) < 8:
            polished.append(para)
            continue
        sents = get_sentences(para)
        if len(sents) >= 3 and burstiness_score(sents) < 0.28:
            p = polish_paragraph(para)
            p = expand_contractions(p)
            p = apply_ai_phrase_cleanup(p)
            polished.append(p)
            ran_polish = True
        else:
            polished.append(enforce_sentence_variety(para))

    draft = "\n\n".join(polished)

    # Phase 6: Measure overall burstiness for UI display
    all_sentences = get_sentences(draft)
    score = burstiness_score(all_sentences)

    # Phase 7: Truncate to word limit
    draft = truncate_to_word_limit(draft, target_word_limit)
    final_count = count_words(draft)

    return JSONResponse({
        "original_count": original_count,
        "final_count": final_count,
        "humanized_text": draft,
        "burstiness_score": round(score, 3),
        "polish_applied": ran_polish,
    })
