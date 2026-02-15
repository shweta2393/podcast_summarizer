"""Text summarization using HuggingFace BART model directly."""

import math
import re
from collections import Counter
from transformers import BartForConditionalGeneration, BartTokenizer

_model = None
_tokenizer = None

# Maximum tokens the model can process at once
_MAX_CHUNK_TOKENS = 900
_SUMMARY_MAX_LENGTH = 180
_SUMMARY_MIN_LENGTH = 40
_MODEL_NAME = "facebook/bart-large-cnn"

# Common English stop words to ignore during scoring
_STOP_WORDS = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o",
    "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn", "also", "got", "get",
    "like", "know", "going", "go", "well", "right", "yeah", "okay",
    "oh", "uh", "um", "really", "actually", "basically", "literally",
    "thing", "things", "said", "says", "say", "would", "could",
})


# ---------------------------------------------------------------------------
# Model loading & BART summarization
# ---------------------------------------------------------------------------

def _load_model():
    global _model, _tokenizer
    if _model is None:
        print("Loading summarization model...")
        _tokenizer = BartTokenizer.from_pretrained(_MODEL_NAME)
        _model = BartForConditionalGeneration.from_pretrained(_MODEL_NAME)
        _model.eval()
        print("Summarization model loaded.")
    return _model, _tokenizer


def _bart_summarize(text: str, max_length: int = 150, min_length: int = 40) -> str:
    """Run BART summarization on a single text string."""
    model, tokenizer = _load_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Content type detection
# ---------------------------------------------------------------------------

def _detect_content_type(text: str) -> str:
    """
    Detect whether the transcript is a song/lyrics or speech/podcast.
    Returns 'song' or 'speech'.
    """
    words = text.lower().split()
    total_words = len(words)

    if total_words == 0:
        return "speech"

    # Heuristic 1: Repetition ratio
    ngrams = []
    for i in range(len(words) - 3):
        ngrams.append(" ".join(words[i:i + 4]))
    ngram_counts = Counter(ngrams)
    repeated = sum(1 for _, c in ngram_counts.items() if c >= 2)
    repetition_ratio = repeated / max(len(ngram_counts), 1)

    # Heuristic 2: Song vocabulary
    song_words = {
        "love", "heart", "baby", "feel", "night", "dream", "soul",
        "dance", "sing", "song", "music", "rhythm", "beat", "melody",
        "yeah", "oh", "ooh", "la", "na", "hey", "whoa",
        "forever", "fire", "light", "sky", "rain", "tears",
        "hold", "kiss", "touch", "body", "eyes", "alone",
        "broken", "falling", "lost", "crazy", "beautiful",
    }
    song_word_hits = sum(1 for w in words if w in song_words)
    song_word_ratio = song_word_hits / total_words

    # Heuristic 3: Transcript length
    is_short = total_words < 800

    # Heuristic 4: Sentence structure
    sentence_endings = len(re.findall(r'[.!?]', text))
    endings_per_100_words = (sentence_endings / total_words) * 100

    # Score
    score = 0
    if repetition_ratio > 0.15:
        score += 3
    elif repetition_ratio > 0.08:
        score += 1
    if song_word_ratio > 0.06:
        score += 2
    elif song_word_ratio > 0.03:
        score += 1
    if is_short:
        score += 1
    if endings_per_100_words < 1.5:
        score += 1

    content_type = "song" if score >= 4 else "speech"
    print(f"Content type detected: {content_type} "
          f"(score={score}, repetition={repetition_ratio:.2f}, "
          f"song_words={song_word_ratio:.3f}, length={total_words}w, "
          f"endings={endings_per_100_words:.1f}/100w)")
    return content_type


# ---------------------------------------------------------------------------
# Text chunking & summarization
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_words: int = _MAX_CHUNK_TOKENS) -> list[str]:
    """Split text into chunks that fit within the model's token limit."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)
    return chunks


def _summarize_text(text: str, content_type: str) -> str:
    """Summarize a piece of text, handling long inputs by chunking."""
    words = text.split()

    if len(words) < 50:
        return text

    chunks = _chunk_text(text)
    summaries = []

    for chunk in chunks:
        chunk_words = chunk.split()
        if len(chunk_words) < 30:
            summaries.append(chunk)
            continue

        max_len = min(_SUMMARY_MAX_LENGTH, max(len(chunk_words) // 2, 50))
        min_len = min(_SUMMARY_MIN_LENGTH, max_len - 10)

        result = _bart_summarize(chunk, max_length=max_len, min_length=min_len)
        summaries.append(result)

    combined = " ".join(summaries)

    # If we had multiple chunks, do a final summarization pass
    if len(chunks) > 1 and len(combined.split()) > 100:
        try:
            combined = _bart_summarize(combined, max_length=300, min_length=80)
        except Exception:
            pass

    return combined


# ---------------------------------------------------------------------------
# Sentence / line splitting
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into complete sentences.
    Only keeps sentences that end with proper punctuation.
    """
    # Split on sentence-ending punctuation followed by whitespace or quote
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) < 15:
            continue
        # Must end with sentence-ending punctuation to be "complete"
        if re.search(r'[.!?]["\'"]?\s*$', s):
            sentences.append(s)
    return sentences


def _split_into_lines(text: str) -> list[str]:
    """Split song transcript into lyrical lines/phrases."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    lines = []
    for part in parts:
        sub = re.split(
            r',\s+|\s+and\s+|\s+but\s+|\s+cause\s+|\s+\'cause\s+|\s+so\s+(?=[A-Z])',
            part,
        )
        for s in sub:
            s = s.strip()
            if len(s) > 5:
                lines.append(s)
    return lines


# ---------------------------------------------------------------------------
# Word importance scoring (TF-IDF inspired)
# ---------------------------------------------------------------------------

def _compute_word_scores(sentences: list[str]) -> dict[str, float]:
    """
    Compute a TF-IDF-like importance score for each word across sentences.
    Words that appear in some sentences but not all are more distinctive.
    """
    num_sentences = len(sentences)
    if num_sentences == 0:
        return {}

    # Document frequency: how many sentences contain each word
    doc_freq: Counter = Counter()
    for s in sentences:
        unique_words = set(w.lower() for w in re.findall(r'[a-z]+', s.lower()))
        for w in unique_words:
            doc_freq[w] += 1

    # IDF-like score: words in fewer sentences are more distinctive
    word_scores = {}
    for word, df in doc_freq.items():
        if word in _STOP_WORDS:
            continue
        if len(word) < 3:
            continue
        # Standard IDF formula
        word_scores[word] = math.log((num_sentences + 1) / (df + 1)) + 1

    return word_scores


def _score_sentence(
    sentence: str,
    word_scores: dict[str, float],
    position: int,
    total_sentences: int,
) -> float:
    """
    Score a single sentence for informativeness.

    Factors:
    - Word importance (TF-IDF)
    - Sentence position (beginning/end of transcript get a boost)
    - Sentence length (prefer moderate length — not too short, not too long)
    - Completeness (must end with punctuation)
    """
    words = [w.lower() for w in re.findall(r'[a-z]+', sentence.lower())]
    word_count = len(words)

    if word_count < 6:
        return 0.0

    # 1. Word importance: average importance of content words
    content_scores = [word_scores.get(w, 0) for w in words if w not in _STOP_WORDS]
    if not content_scores:
        return 0.0
    avg_word_importance = sum(content_scores) / len(content_scores)

    # 2. Position bonus: first 15% and last 10% of sentences are often key
    pos_ratio = position / max(total_sentences - 1, 1)
    position_bonus = 0.0
    if pos_ratio < 0.15:
        position_bonus = 1.5  # Introductory context
    elif pos_ratio > 0.90:
        position_bonus = 1.0  # Closing remarks / conclusions

    # 3. Length preference: moderate-length sentences (12-35 words) score highest
    if 12 <= word_count <= 35:
        length_bonus = 1.0
    elif 8 <= word_count <= 45:
        length_bonus = 0.5
    else:
        length_bonus = 0.0

    # 4. Content density: ratio of non-stop words
    content_ratio = len(content_scores) / max(word_count, 1)
    density_bonus = content_ratio * 2.0

    return avg_word_importance + position_bonus + length_bonus + density_bonus


# ---------------------------------------------------------------------------
# Key points extraction — Speech / Podcast
# ---------------------------------------------------------------------------

def _extract_key_points_speech(text: str) -> list[str]:
    """
    Extract key points by scoring every complete sentence for informativeness,
    then picking the top diverse set.
    """
    sentences = _split_into_sentences(text)

    if len(sentences) <= 5:
        return sentences if sentences else []

    # Score each sentence
    word_scores = _compute_word_scores(sentences)
    scored = []
    for i, s in enumerate(sentences):
        score = _score_sentence(s, word_scores, i, len(sentences))
        if score > 0:
            scored.append((score, i, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Pick top sentences ensuring diversity and ordering by position
    selected = []
    for score, pos, sentence in scored:
        if len(selected) >= 6:
            break
        # Ensure not too similar to already-selected sentences
        if any(_similar(sentence, sel_s) for _, sel_s in selected):
            continue
        selected.append((pos, sentence))

    # Sort by original position so key points follow the flow of the podcast
    selected.sort(key=lambda x: x[0])

    return [s for _, s in selected]


# ---------------------------------------------------------------------------
# Key points extraction — Song
# ---------------------------------------------------------------------------

def _extract_key_points_song(text: str) -> list[str]:
    """
    Extract key points from a song.
    For songs: main themes, chorus/hook, and recurring motifs.
    """
    key_points = []
    words_lower = text.lower().split()

    # 1. Detect chorus / hook (most repeated phrases)
    for n in (6, 5, 4):
        ngrams = []
        for i in range(len(words_lower) - n + 1):
            ngrams.append(" ".join(words_lower[i:i + n]))
        counts = Counter(ngrams)
        for phrase, count in counts.most_common(5):
            if count >= 2:
                original = _find_original_case(text, phrase)
                entry = f'Recurring hook: "{original}" (appears {count}x)'
                if not any(_similar(entry, kp) for kp in key_points):
                    key_points.append(entry)
                    break
        if len(key_points) >= 2:
            break

    # 2. Identify themes
    theme_groups = {
        "Love & Romance": ["love", "heart", "kiss", "romance", "darling", "baby", "forever", "together"],
        "Loss & Longing": ["miss", "gone", "lost", "alone", "without", "memory", "remember", "forget"],
        "Hope & Dreams": ["dream", "hope", "wish", "believe", "tomorrow", "someday", "future", "light"],
        "Strength & Empowerment": ["strong", "fight", "rise", "power", "stand", "brave", "free", "freedom"],
        "Heartbreak & Pain": ["broken", "tears", "cry", "pain", "hurt", "sad", "falling", "apart"],
        "Joy & Celebration": ["happy", "dance", "celebrate", "party", "alive", "fun", "joy", "smile"],
        "Night & Mystery": ["night", "dark", "shadow", "moon", "stars", "midnight", "secret", "silence"],
        "Nature & Journey": ["road", "rain", "river", "mountain", "sky", "sea", "wind", "sun", "fly"],
    }
    word_set = set(words_lower)
    theme_scores = []
    for theme_name, theme_words in theme_groups.items():
        hits = sum(1 for w in theme_words if w in word_set)
        freq = sum(words_lower.count(w) for w in theme_words)
        if hits >= 2 or freq >= 3:
            theme_scores.append((freq, theme_name, hits))

    theme_scores.sort(reverse=True)
    for freq, theme_name, hits in theme_scores[:3]:
        key_points.append(f"Theme: {theme_name}")

    # 3. Emotional tone
    positive_words = {"love", "happy", "smile", "beautiful", "wonderful", "joy", "light", "dream", "alive", "free"}
    negative_words = {"broken", "cry", "tears", "pain", "lost", "dark", "alone", "hurt", "sad", "fear"}
    pos_count = sum(1 for w in words_lower if w in positive_words)
    neg_count = sum(1 for w in words_lower if w in negative_words)
    if pos_count > neg_count and pos_count >= 2:
        key_points.append("Emotional tone: Uplifting / Positive")
    elif neg_count > pos_count and neg_count >= 2:
        key_points.append("Emotional tone: Melancholic / Reflective")
    elif pos_count >= 1 and neg_count >= 1:
        key_points.append("Emotional tone: Mixed / Bittersweet")

    # 4. Song structure observation
    lines = _split_into_lines(text)
    if lines:
        key_points.append(f"Structure: ~{len(lines)} lyrical lines identified")

    return key_points[:7]


def _find_original_case(text: str, lowercase_phrase: str) -> str:
    """Find the original-case version of a lowercase phrase in text."""
    text_lower = text.lower()
    idx = text_lower.find(lowercase_phrase)
    if idx >= 0:
        return text[idx:idx + len(lowercase_phrase)]
    return lowercase_phrase


# ---------------------------------------------------------------------------
# Highlights extraction — Speech / Podcast
# ---------------------------------------------------------------------------

def _extract_highlights_speech(text: str) -> list[str]:
    """
    Extract the most notable quotes / statements from a podcast.
    Uses word-importance scoring plus topical signal words.
    Only selects complete sentences.
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    word_scores = _compute_word_scores(sentences)

    # Topical signal words that indicate a "quotable" statement
    signal_words = [
        "important", "key", "significant", "interesting", "surprising",
        "believe", "think", "opinion", "personally", "argue",
        "challenge", "solution", "problem", "opportunity", "risk",
        "future", "change", "transform", "revolution", "trend",
        "discover", "learn", "insight", "lesson", "realize",
        "advice", "recommend", "suggest", "should", "must",
        "amazing", "incredible", "remarkable", "extraordinary",
        "never", "always", "everyone", "nobody", "everything",
        "secret", "truth", "mistake", "success", "failure",
    ]

    scored = []
    for i, s in enumerate(sentences):
        s = s.strip()
        words = s.lower().split()
        word_count = len(words)

        if word_count < 8 or word_count > 50:
            continue

        # Base: word importance score
        base_score = _score_sentence(s, word_scores, i, len(sentences))

        # Signal word bonus
        signal_bonus = 0
        for w in signal_words:
            if w in words:
                signal_bonus += 3

        # Quotation marks suggest someone said something notable
        if '"' in s or "'" in s:
            signal_bonus += 4

        # First-person statements often carry opinions/insights
        if any(s.lower().startswith(p) for p in ["i think", "i believe", "in my", "my advice"]):
            signal_bonus += 3

        total_score = base_score + signal_bonus
        scored.append((total_score, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    highlights = []
    for _, sentence in scored:
        if len(highlights) >= 5:
            break
        if not any(_similar(sentence, h) for h in highlights):
            highlights.append(sentence)

    return highlights


# ---------------------------------------------------------------------------
# Highlights extraction — Song
# ---------------------------------------------------------------------------

def _extract_highlights_song(text: str) -> list[str]:
    """Extract the most memorable / poetic lines from a song."""
    lines = _split_into_lines(text)
    if not lines:
        return []

    lyric_signal_words = [
        "love", "heart", "soul", "feel", "feeling", "pain", "tears",
        "cry", "smile", "happy", "dream", "hope", "wish",
        "fire", "flame", "light", "dark", "shadow", "moon", "stars",
        "sky", "rain", "ocean", "river", "sun", "night", "dawn",
        "never", "forever", "always", "every", "only", "everything",
        "nothing", "nobody", "somebody", "anymore",
        "fly", "fall", "rise", "burn", "shine", "fade", "break",
        "breathe", "whisper", "scream", "running", "falling",
        "beautiful", "wonderful", "incredible",
    ]

    scored = []
    for line in lines:
        line = line.strip()
        words = line.lower().split()
        word_count = len(words)
        if word_count < 4 or word_count > 40:
            continue

        score = 0
        for w in lyric_signal_words:
            if w in words:
                score += 4
        if 6 <= word_count <= 20:
            score += 3

        contrast_pairs = [
            ("love", "hate"), ("light", "dark"), ("fire", "ice"),
            ("rise", "fall"), ("lost", "found"), ("break", "heal"),
            ("day", "night"), ("laugh", "cry"), ("hello", "goodbye"),
        ]
        for a, b in contrast_pairs:
            if a in words and b in words:
                score += 6
        if "!" in line:
            score += 2
        if "?" in line:
            score += 2

        filler = ["yeah yeah", "la la", "na na", "oh oh", "uh huh"]
        for f in filler:
            if f in line.lower():
                score -= 5

        if score > 0:
            scored.append((score, line))

    scored.sort(key=lambda x: x[0], reverse=True)

    highlights = []
    for _, line in scored:
        if len(highlights) >= 5:
            break
        if not any(_similar(line, h) for h in highlights):
            highlights.append(line)

    return highlights


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _similar(a: str, b: str) -> bool:
    """Check if two strings are too similar (basic overlap check)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
    return overlap > 0.6


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_summary(transcript: str) -> dict:
    """
    Generate a full summary, key points, and highlights from a transcript.
    Automatically detects whether content is a song or speech and adapts.

    Returns:
        dict with keys: summary, key_points, highlights, content_type
    """
    if not transcript or len(transcript.strip()) < 20:
        return {
            "summary": "Transcript too short to summarize.",
            "key_points": [],
            "highlights": [],
            "content_type": "speech",
        }

    content_type = _detect_content_type(transcript)

    print("Generating summary...")
    summary = _summarize_text(transcript, content_type)

    print("Extracting key points...")
    if content_type == "song":
        key_points = _extract_key_points_song(transcript)
    else:
        key_points = _extract_key_points_speech(transcript)

    print("Extracting highlights...")
    if content_type == "song":
        highlights = _extract_highlights_song(transcript)
    else:
        highlights = _extract_highlights_speech(transcript)

    return {
        "summary": summary,
        "key_points": key_points,
        "highlights": highlights,
        "content_type": content_type,
    }
