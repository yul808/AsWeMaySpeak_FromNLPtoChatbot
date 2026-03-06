# ChatGPT was used for consulting purposes
# still some major issues that i couldnt help to solve yet
# game is mostly playable (fundamental concepts regarding SpaCy work) but not yet beatable

import spacy
import numpy as np
import random

# Loading en_core_web_lg
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    raise SystemExit


def cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def random_vector_word(vocab):
    candidates = [
        lex for lex in vocab
        if lex.is_alpha and lex.has_vector and len(lex.text) >= 3
    ]
    return random.choice(candidates)


def similarity(w1, w2):
    t1, t2 = nlp(w1)[0], nlp(w2)[0]
    if not t1.has_vector or not t2.has_vector:
        return 0.0
    return cosine(t1.vector, t2.vector)


def suggest_words(current_guess, target, threshold):

    guess_vec = nlp(current_guess)[0].vector
    target_vec = nlp(target)[0].vector

    guess_to_target = cosine(guess_vec, target_vec)

    candidates = []
    for lex in nlp.vocab:
        if lex.is_alpha and lex.has_vector and len(lex.text) >= 3:
            sim_target = cosine(target_vec, lex.vector)
            candidates.append((lex.text, sim_target))

    # Sorting by similarity to target
    candidates.sort(key=lambda x: x[1], reverse=True)

    much_better = [w for w, s in candidates
                   if guess_to_target + 0.15 < s < threshold - 0.02]

    slightly_better = [w for w, s in candidates
                       if guess_to_target + 0.02 < s < guess_to_target + 0.10]

    worse = [w for w, s in candidates
             if s < guess_to_target - 0.05]

    # fallback strategies to avoid faulty hints
    def safe_choice(lst, fallback_index):
        if lst:
            return random.choice(lst)
        return candidates[fallback_index][0] #problem:IndexError occurs with some words (especially when using suggested words); dont know why that happens yet

    much_better_w = safe_choice(much_better, 5)
    slightly_better_w = safe_choice(slightly_better, 200)
    worse_w = safe_choice(worse, -50)

    return much_better_w, slightly_better_w, worse_w # problem: the hints dont really lead to the final word yet


# GAME below

def play_game(threshold=0.85):
    print("Guess the hidden word! (type 'exit' to quit)\n")

    target = random_vector_word(nlp.vocab).text
    guesses = []

    while True:
        guess = input("Your guess: ").strip().lower()
        if guess == "exit":
            print("\nGame ended. The word was:", target)
            return

        guesses.append(guess)

        sim = similarity(guess, target)

        # feedback
        print(f"\nYou guessed: {guess}")
        print(f"Similarity value: {sim:.3f}")
        print("Guesses so far:", ", ".join(guesses))

        # suggestions
        close, medium, unrelated = suggest_words(guess, target, threshold)
        print("\nHere are some hints:")
        print(" - Try something like:", close)
        print(" - Or maybe:", medium)
        print(" - Definitely not like:", unrelated)

        # check threshold
        if sim >= threshold:
            print("\nYou're VERY close! You get ONE final guess.\n")
            final_guess = input("Final guess: ").strip().lower()
            guesses.append(final_guess)

            print("\nFinal guesses:", ", ".join(guesses))
            print("Game over!")
            print("The word was:", target)
            return


if __name__ == "__main__":
    play_game()