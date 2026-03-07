"""
The NLP Self-Publisher | Rewrite your Favorite Books
===============
A word-guessing game powered by GloVe word embeddings and Project Gutenberg texts.

How it works:
  1. The game picks 6 random "target" words (the red herrings), revealed one at a time.
  2. For each target word, the player guesses a semantically close word.
  3. Cosine similarity (via GloVe vectors) measures how close the guess is.
  4. MISS guesses (below NEAR_THRESHOLD) are added to the story but the player
     keeps trying until they get a HIT or WARM — advancing to the next round.
  5. Every guess (including all misses) contributes sentences to the story,
     so wrong guesses create delightfully weird, nonsensical passages.
  6. When all 6 rounds are done the full story is printed.



GloVe:
  Expects wiki_giga_2024_200_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt
  to be in the same directory as main.py

  Download the GloVe vectors file from the link in the repository README.md
  Place it in the same directory as `main.py`.


Gutenberg texts:
  3 Books are provided via the git.
  Place any other book you like as plain text files (e.g. from Project Gutenberg) in ./gutenberg_texts/
  The script ships with a tiny fallback corpus for testing.
"""

import os
import math
import random
import zipfile
import urllib.request
import re

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

GLOVE_DIR        = "."                     # same directory as main.py
GLOVE_FILE       = "wiki_giga_2024_200_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"
GLOVE_DOWNLOAD   = False

GUTENBERG_DIR    = "./gutenberg_texts"     # folder with all the .txt files
NUM_TARGET_WORDS = 6                       # anchor words per game
SENTENCES_PER_GUESS = 2                    # story sentences generated per guess

# Similarity thresholds
HIT_THRESHOLD    = 0.65   # cosine sim ≥ this  → "close" (green)
NEAR_THRESHOLD   = 0.45   # cosine sim ≥ this  → "warm"  (yellow)
# below NEAR_THRESHOLD   → "miss"               (red) — still in story!

# Words to draw target words from (a curated, story-friendly list)
TARGET_WORD_POOL = [
    "ocean", "mirror", "clock", "forest", "lantern", "shadow", "bridge",
    "letter", "candle", "garden", "storm", "window", "silence", "river",
    "mountain", "dream", "smoke", "door", "fire", "moon", "rain", "ship",
    "tower", "road", "voice", "stone", "bird", "flower", "winter", "key",
    "library", "secret", "mask", "journey", "light", "ghost", "tear",
    "sword", "rose", "cloud", "echo", "wave", "sun", "star", "night",
]

# Fallback mini-corpus (used when no Gutenberg texts are found)
FALLBACK_SENTENCES = [
    "The old clock ticked quietly in the empty room.",
    "She walked across the bridge without looking back.",
    "A letter arrived that changed everything.",
    "The forest was dark, and full of whispers.",
    "He held the candle close to the ancient map.",
    "Rain tapped gently against the window pane.",
    "The ship disappeared beyond the horizon at dawn.",
    "A voice echoed through the narrow stone corridor.",
    "The garden had been forgotten for many years.",
    "She found the key beneath a loose floorboard.",
    "Smoke rose slowly from the chimney of the cottage.",
    "The shadow on the wall moved, though no one did.",
    "Stars reflected in the still surface of the river.",
    "He opened the door and stepped into the unknown.",
    "The mountain stood silent under a blanket of snow.",
    "A single bird perched on the highest tower.",
    "The rose had lost its petals long ago.",
    "Fire crackled and spat in the great stone hearth.",
    "A secret was buried with her in the garden.",
    "The mirror showed a face that was not her own.",
    "Every road leads somewhere, even the ones we fear.",
    "The library held books that no one had opened in decades.",
    "She dreamed of the ocean every night without fail.",
    "A ghost wandered the halls long after the others had left.",
    "The lantern swung in the wind above the empty street.",
    "Winter came early and stayed longer than expected.",
    "He carried the stone in his pocket as a reminder.",
    "The wave broke and pulled the sand back to the sea.",
    "Light filtered through the leaves in golden threads.",
    "The sword rested in its scabbard, waiting for a war.",
]


# ──────────────────────────────────────────────────────────────────────────────
# GLOVE LOADING
# ──────────────────────────────────────────────────────────────────────────────

def _download_glove():
    """Download GloVe 6B 50d vectors if not present."""
    os.makedirs(GLOVE_DIR, exist_ok=True)
    dest = os.path.join(GLOVE_DIR, GLOVE_FILE)
    if os.path.exists(dest):
        return dest

    zip_path = os.path.join(GLOVE_DIR, "glove.6B.zip")
    if not os.path.exists(zip_path):
        print("Downloading GloVe vectors (~170 MB) — this happens once …")
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")

    print("Extracting 50d vectors …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(GLOVE_FILE, GLOVE_DIR)
    print("Extraction done.")
    return dest


def load_glove(vocab=None):
    """
    Load GloVe embeddings into a dict  {word: [float, …]}.
    If `vocab` is provided (a set of strings) only those words are loaded
    to save memory.
    """
    path = os.path.join(GLOVE_DIR, GLOVE_FILE)
    if not os.path.exists(path):
        if GLOVE_DOWNLOAD:
            path = _download_glove()
        else:
            raise FileNotFoundError(
                f"GloVe file not found at {path}. "
                "Set GLOVE_DOWNLOAD=True or download manually."
            )

    print("Loading GloVe vectors …  THIS MIGHT TAKE A WHILE", end=" ", flush=True)
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word  = parts[0]
            if vocab and word not in vocab:
                continue
            embeddings[word] = list(map(float, parts[1:]))
    print(f"loaded {len(embeddings):,} words.")
    return embeddings


# ──────────────────────────────────────────────────────────────────────────────
# VECTOR MATH
# ──────────────────────────────────────────────────────────────────────────────

def cosine_similarity(v1, v2):
    """Cosine similarity between two vectors (lists of floats)."""
    dot   = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def similarity(word_a, word_b, embeddings):
    """Return cosine similarity, or None if either word is not in embeddings."""
    va = embeddings.get(word_a.lower())
    vb = embeddings.get(word_b.lower())
    if va is None or vb is None:
        return None
    return cosine_similarity(va, vb)


def classify_guess(sim_score):
    """Return a (label, colour_code) tuple for the similarity score."""
    if sim_score >= HIT_THRESHOLD:
        return "HIT 🟢",   "\033[92m"   # green
    elif sim_score >= NEAR_THRESHOLD:
        return "WARM 🟡",  "\033[93m"   # yellow
    else:
        return "MISS 🔴",  "\033[91m"   # red


# ──────────────────────────────────────────────────────────────────────────────
# TERMINAL COLOURS
# ──────────────────────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
CYAN  = "\033[96m"
WHITE = "\033[97m"


# ──────────────────────────────────────────────────────────────────────────────
# GUTENBERG CORPUS
# ──────────────────────────────────────────────────────────────────────────────

# Transition phrases inserted between story sentences.
# Grouped by mood so they can be varied across the story.
TRANSITIONS = [
    "And yet,",
    "But then,",
    "Meanwhile,",
    "Suddenly,",
    "In that moment,",
    "Without warning,",
    "Not long after,",
    "As if in answer,",
    "Still,",
    "Even so,",
    "Before long,",
    "At last,",
]


def load_books():
    """
    Scan GUTENBERG_DIR and return a dict of {display_name: filepath}.
    Falls back to the built-in corpus if no files are found.
    """
    books = {}
    if os.path.isdir(GUTENBERG_DIR):
        for fname in sorted(os.listdir(GUTENBERG_DIR)):
            if not fname.startswith("."):
                full_path = os.path.join(GUTENBERG_DIR, fname)
                if os.path.isfile(full_path):
                    books[fname] = full_path
    return books


def pick_book():
    """
    Asks the player to choose a book. Returns (book_name, [sentences]).
    Falls back to FALLBACK_SENTENCES if no books are available.
    """
    books = load_books()

    if not books:
        print(f"  {DIM}(No Gutenberg texts found — using built-in fallback corpus.){RESET}")
        return "fallback", FALLBACK_SENTENCES[:]

    names = list(books.keys())
    print(f"\n{BOLD}Choose your book — the story will be written in its voice:{RESET}\n")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    print()

    while True:
        choice = input(f"  Enter number (1–{len(names)}) → ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            chosen_name = names[int(choice) - 1]
            break
        print(f"  {DIM}Please enter a number between 1 and {len(names)}.{RESET}")

    # Parse sentences from the chosen file
    sentences = []
    with open(books[chosen_name], "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    raw = re.split(r'(?<=[.!?])\s+', text)
    for s in raw:
        s = s.strip().replace("\n", " ")
        if 30 < len(s) < 300:
            sentences.append(s)

    if not sentences:
        print(f"  {DIM}Could not parse sentences from that file — using fallback.{RESET}")
        return chosen_name, FALLBACK_SENTENCES[:]

    print(f"\n  {DIM}Loaded {len(sentences):,} sentences from '{chosen_name}'.{RESET}\n")
    return chosen_name, sentences


def find_relevant_sentences(word, sentences, embeddings, n=SENTENCES_PER_GUESS):
    """
    Returns up to `n` sentences whose words are most semantically close
    to `word` on average. Falls back to random selection if needed.
    """
    word_vec = embeddings.get(word.lower())
    if word_vec is None:
        return random.sample(sentences, min(n, len(sentences)))

    scored = []
    for sent in sentences:
        words_in_sent = re.findall(r"[a-z']+", sent.lower())
        sims = [
            cosine_similarity(word_vec, embeddings[w])
            for w in words_in_sent
            if w in embeddings
        ]
        if sims:
            scored.append((sum(sims) / len(sims), sent))

    if not scored:
        return random.sample(sentences, min(n, len(sentences)))

    scored.sort(key=lambda x: x[0], reverse=True)
    # Pick from top-20 pool with some randomness so stories vary
    pool = scored[:20]
    chosen = random.sample(pool, min(n, len(pool)))
    return [s for _, s in chosen]


def build_story(all_rounds):
    """
    Flatten all collected sentences from every round into one continuous story,
    inserting a varied transition phrase between each sentence for flow.
    The first sentence never gets a transition prefix.
    """
    # Collect all sentences in order, across all rounds and all attempts
    all_sentences = []
    for _target, round_story, _final_guess, _final_sim in all_rounds:
        for _guess, _sim, sents in round_story:
            all_sentences.extend(sents)

    if not all_sentences:
        return ""

    # Shuffle transitions so they don't repeat predictably
    transition_pool = TRANSITIONS[:]
    random.shuffle(transition_pool)
    transition_cycle = (transition_pool * ((len(all_sentences) // len(transition_pool)) + 2))

    story_lines = [all_sentences[0]]  # first sentence — no transition
    for i, sent in enumerate(all_sentences[1:], 0):
        transition = transition_cycle[i]
        # Lower-case the first letter of the sentence after a transition
        joined = f"{transition} {sent[0].lower()}{sent[1:]}" if sent else transition
        story_lines.append(joined)

    return " ".join(story_lines)


# ──────────────────────────────────────────────────────────────────────────────
# GAME LOOP
# ──────────────────────────────────────────────────────────────────────────────


def print_banner():
    print(f"\n{BOLD}{CYAN}{'═'*55}")
    print("  The NLP Self-Publisher | Rewrite your Favorite Books   ")
    print(f"{'═'*55}{RESET}\n")
    print("  Six mystery words will be revealed one at a time.")
    print("  Guess a semantically close word for each one.")
    print("  Misses are added to your story — but you keep")
    print("  trying until you get a HIT 🟢 or WARM 🟡.\n")
    print(f"  Hit  🟢  sim ≥ {HIT_THRESHOLD:.2f}   Warm 🟡  sim ≥ {NEAR_THRESHOLD:.2f}   Miss 🔴\n")


def play_round(round_num, target_word, sentences, embeddings):
    """
    Plays one guessing round.

    The player keeps guessing until they achieve HIT or WARM.
    Every MISS is added to the story immediately (as a weird fragment),
    then the player is prompted again for the same target word.

    """
    print(f"{BOLD}─── Round {round_num} / {NUM_TARGET_WORDS} ───{RESET}")
    print(f"  Target word:  {BOLD}{WHITE}{target_word.upper()}{RESET}\n")

    round_story = []   # collects (guess, sim, sents) for every attempt

    while True:
        guess = input("  Your guess → ").strip().lower()
        if not guess:
            continue

        sim = similarity(target_word, guess, embeddings)

        if sim is None:
            print(f"  {DIM}⚠  '{guess}' not found in vocabulary — try another word.{RESET}\n")
            continue

        label, colour = classify_guess(sim)
        print(f"  {colour}{label}{RESET}  similarity: {colour}{sim:.3f}{RESET}")

        # Always generate story sentences for this guess
        sents = find_relevant_sentences(guess, sentences, embeddings)
        round_story.append((guess, sim, sents))

        if sim < NEAR_THRESHOLD:
            # MISS — add to story but keep going
            print(f"  {DIM}Not close enough — keep trying! Your guess is woven into the story.{RESET}\n")
        else:
            # HIT or WARM — round complete
            print()
            break

    final_guess = round_story[-1][0]
    final_sim   = round_story[-1][1]
    return round_story, final_guess, final_sim


def run_game():
    print_banner()

    # ── Load resources ──────────────────────────────────────────────────────
    book_name, sentences = pick_book()
    embeddings = load_glove()  # load full vocabulary — needed for player guesses

    # Filter target pool to words actually in embeddings
    valid_targets = [w for w in TARGET_WORD_POOL if w in embeddings]
    if len(valid_targets) < NUM_TARGET_WORDS:
        raise RuntimeError("Not enough target words found in embeddings. "
                           "Check your GloVe file.")

    # ── Pick target words (hidden from player until each round starts) ───────
    target_words = random.sample(valid_targets, NUM_TARGET_WORDS)

    print(f"{BOLD}Get ready — {NUM_TARGET_WORDS} words will be revealed one at a time.{RESET}")
    print(f"{DIM}Press Enter to begin …{RESET}")
    input()

    # ── Rounds ───────────────────────────────────────────────────────────────
    # all_rounds: list of (target, round_story, final_guess, final_sim)
    # where round_story = list of (guess, sim, [sents]) for every attempt
    all_rounds = []

    for i, target in enumerate(target_words, 1):
        round_story, final_guess, final_sim = play_round(
            i, target, sentences, embeddings
        )
        all_rounds.append((target, round_story, final_guess, final_sim))

        remaining = NUM_TARGET_WORDS - i
        if remaining > 0:
            print(f"  {DIM}── {remaining} word{'s' if remaining != 1 else ''} remaining. Press Enter to continue …{RESET}")
            input()

    # ── Build and print the unified story ────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'═'*55}")
    print("   ✨  Y O U R   S T O R Y")
    print(f"   {DIM}written in the voice of: {book_name}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*55}{RESET}\n")

    story = build_story(all_rounds)
    # Word-wrap the story at ~80 chars for readability
    words = story.split()
    line, lines = [], []
    for word in words:
        line.append(word)
        if len(" ".join(line)) > 80:
            lines.append(" ".join(line[:-1]))
            line = [word]
    if line:
        lines.append(" ".join(line))
    print("  " + "\n  ".join(lines))

    print(f"\n{BOLD}{CYAN}{'═'*55}{RESET}\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    hits          = sum(1 for _, _, _, s in all_rounds if s >= HIT_THRESHOLD)
    warms         = sum(1 for _, _, _, s in all_rounds if NEAR_THRESHOLD <= s < HIT_THRESHOLD)
    total_guesses = sum(len(rs) for _, rs, _, _ in all_rounds)
    print(f"  Final score:  🟢 {hits} hits   🟡 {warms} warm")
    print(f"  Total guesses made: {total_guesses}   (across {NUM_TARGET_WORDS} rounds)\n")

    again = input("  Play again? (y / n) → ").strip().lower()
    if again == "y":
        run_game()
    else:
        print(f"\n{BOLD}Thanks for playing! Your story has been written.{RESET}\n")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_game()
