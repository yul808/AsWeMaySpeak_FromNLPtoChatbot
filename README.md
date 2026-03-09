# The NLP Self-Publisher: Rewrite your Favorite Books
This repository contains my final project (as well as some in-class assignments) for our course "As We May Speak: From NLP to Chatbot" in the "Practical Experience in Digital Media 2" module. In the course we have looked at different forms of natural language processing prior to the popularization of Large Language Models. The project at hand was made using some of the techniques learned in the course. 

## 1. The Concept
You are about to play a small game that uses Natural Language Processing (NLP) to let you rewrite your favorite books through some word guessing. 

How it works:
  1. The game picks 6 random "target" words (the red herrings), revealed one at a time.
  2. For each target word, the player guesses a semantically close word.
  3. Cosine similarity (via GloVe vectors) measures how close the guess is.
  4. MISS guesses (below NEAR_THRESHOLD) are added to the story but the player
     keeps trying until they get a HIT or WARM — advancing to the next round.
  5. Every guess (including all misses) contributes sentences to the story,
     so wrong guesses create delightfully weird, nonsensical passages.
  6. When all 6 rounds are done the full story is printed.

The project idea is mainly based on a previous in-class assignment (see assignment3.py). Back in class, the approach chosen was insufficient, so subsequently, the program would not amount to any proper output. This new interpretation of the game uses a different approach after having looked at more class material. It also combines the old word guessing aspect with a new conceptual frame, with that frame being the text production aspect that follows the vectoring of GloVe

## 2. Limitations


## 3. The Process


[Initial Presentation NLP.pdf](https://github.com/user-attachments/files/25851983/Initial.Presentation.NLP.pdf)

<img width="634" height="334" alt="Bildschirmfoto 2026-01-29 um 11 22 42" src="https://github.com/user-attachments/assets/dbe2b7f7-7cea-445e-9c7b-8ac46f2fd648" />

<img width="744" height="365" alt="Bildschirmfoto 2026-01-29 um 11 39 00" src="https://github.com/user-attachments/assets/1d40f4e5-d6b2-4ef3-b2f4-e24ee17452a2" />

[Intermediate Presentation NLP .pdf](https://github.com/user-attachments/files/25851882/Intermediate.Presentation.NLP.pdf)

## 4. The Game

### 4.1 Instructions on Running the Code
Before continuing 

### 4.2 Instructions on Playing the Game

Have fun playing. :)



HINT: In case you get stuck and cannot think of any other words to progress, just try repeating goal word. ;)
