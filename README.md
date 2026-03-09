# The NLP Self-Publisher: Rewrite your Favorite Books
This repository contains my final project (as well as some in-class assignments) for our course "As We May Speak: From NLP to Chatbot" in the "Practical Experience in Digital Media 2" module. In the course we have looked at different forms of natural language processing prior to the popularization of Large Language Models. The project at hand was made using some of the techniques learned in the course. 

## 1. The Concept (Short Description)
You are about to play a small game that uses Natural Language Processing (NLP) to let you rewrite your favorite books through some word guessing. 

How it works:
  1. The game picks 6 random "target" words (the red herrings), revealed one at a time.
  2. For each target word, the player guesses a semantically close word.
  3. Cosine similarity (via `GloVe` vectors) measures how close the guess is.
  4. MISS guesses (below `NEAR_THRESHOLD`) are added to the story but the player
     keeps trying until they get a HIT or WARM — advancing to the next round.
  5. Every guess (including all misses) contributes sentences to the story,
     so wrong guesses create delightfully weird, nonsensical passages.
  6. When all 6 rounds are done the full story is printed.


Conceptual Sketch:

<img width="634" height="334" alt="Bildschirmfoto 2026-01-29 um 11 22 42" src="https://github.com/user-attachments/assets/dbe2b7f7-7cea-445e-9c7b-8ac46f2fd648" />

The project idea is mainly based on a previous in-class assignment (see `assignment3.py`). Back in class, the approach chosen was insufficient, so subsequently, the program would not amount to any proper output. This new interpretation of the game uses a different approach after having looked at more class material. It also combines the old word guessing aspect with a new conceptual frame, with that frame being the text production aspect that follows the vectoring of `GloVe`. 

AI transparency: AI (Claude) was used early in the project as a sparring partner to provide a frame work for the code and as a tool to further identify weaknesses. It played a major role in bringing my ideas to life in the development process, as it helped me to understand necessary steps regarding what specific code was needed.

## 2. The Process
Before arriving at this final project, there was another idea that I unfortunately had to scrap due to a lack of proper concept (see Initial Presentation NLP.pdf). Plan B happened to be the reinterpretation of `assignment3.py` that you see here. 

The most difficult part was probably the implementation of the conceptual structure that works in the backround. Having a list of anchor words that has to be semantically compared to the `GloVe` vectors that has to be semantically compared to the smaple texts from the Gutenberg Project took a lot of trial and error (as well as some consultation with AI). The anchor word list turned out much shorter than originally intended, but with little to no cost regarding (re-)playability. 


Backround Structure Sketch:

<img width="744" height="365" alt="Bildschirmfoto 2026-01-29 um 11 39 00" src="https://github.com/user-attachments/assets/1d40f4e5-d6b2-4ef3-b2f4-e24ee17452a2" />


During development, the code as well as the consept both went through a few changes. Initially, the idea was to have the game write independent stories through the player guesses. The provided texts would merely serve as data, that the program would take inspiration from. However, with the texts I had and their embedding, the texts would appear rather chaotic than humorously random. The first step was to add the possibility of transition phrases which at least allowed for a clearer structure in the text. The conceptual solution for that was then to only use one book per play as the data source and to then reframe this aspect of the game as a playful way of rewriting your favorite books. What was nonsensical chaos before is now random text production within a logical frame of 1 particular book and less messy than before. 

Also, the tresholds that defines whether a guess is a hit or a miss (or 'warm') were connected to a lot of trial and error. The goal was to find a point at which finding semantically close words wouldn't be too hard but neither too easy. Right now, the program is at a point where it allows for (sometimes multiple) false guesses, but never makes it too hard to advance. 

[Initial Presentation NLP.pdf](https://github.com/user-attachments/files/25851983/Initial.Presentation.NLP.pdf)

[Intermediate Presentation NLP .pdf](https://github.com/user-attachments/files/25851882/Intermediate.Presentation.NLP.pdf)

## 3. Limitations
Even though, the approach was constantly refined during development as explained above, one should be aware that some boundaries still remain. The most important one being that, the produced texts will not make up for coherent stories. It is more about taking sentences from a source text and looking at what is possible with few gamified rules of engagement. Under this light, the project title might even be seen as a bit misleading. 

Also, the loading times can be quite long at times, taking as long as 1 or 2 minutes at times. This is partly due to the large file size containing the `GloVe` vectors. This is also the reason why the respective file will have to be downloaded seperately, as I couldn't find another solution in that regard. More on that later. 

## 4. The Game

### 4.1 Instructions on Running the Code
To play the game for yourself, first clone the repository. After that, STOP! There are some IMPORTANT steps to follow, as there are some crucial criteria to be met before finally playing. 

1) Download the `GloVe` vectors file here: [https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.200d.zip]. Then place it in the same directory as `main.py` as well as the `gutenberg_texts` foulder. It should be named  `wiki_giga_2024_200_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt `. 


<img width="663" height="257" alt="Bildschirmfoto 2026-03-09 um 23 27 09" src="https://github.com/user-attachments/assets/10af02d5-9e67-4679-b14e-712a68d118ca" />


2) Also check the `gutenberg_texts` foulder. Does it contain your favorite book? :) If not, add it as a `.txt` file. Three books are provided with this repository. You can most likely find your favorite book here: https://www.gutenberg.org 

3) Then run `main.py`. 

### 4.2 Instructions on Playing the Game
1. When playing the game, you will first be asked, which book you want to play with. The program herefore scans the entire  `gutenber_texts ` foulder and will automatically include every `.txt` file. So if at a later point in time you want to use another book you haven't thought of before, you can just easily add it to the foulder.

2. The program will provide you with anchor words. Your semantic goals. You may now take your first guess.

3. The program will now give you feedback regarding your guess. If you got a hit, that's great! If not, try again. You have unlimited tries. Also try rethinking some of the words. E.g. 'dream' can have different meanings: 1) in the context of sleeping; 2) as in goal (in life). Which semantic meaning the vectors descided on, might not necessarily be apparant tight away. (see image)

ALSO NOTE: Due to loading times, give the program some time to print after each guess.

4. Now enjoy your little story. Maybe you can track which sentences were caused by which guess? :)


<img width="627" height="658" alt="Bildschirmfoto 2026-03-07 um 11 24 49" src="https://github.com/user-attachments/assets/11cb2678-7473-481d-9d8e-fab2f5ac3a14" /> 


<img width="627" height="658" alt="Bildschirmfoto 2026-03-07 um 11 27 39" src="https://github.com/user-attachments/assets/8ca04e62-a991-4c2a-be07-64c680daec69" />


<img width="738" height="658" alt="Bildschirmfoto 2026-03-08 um 11 05 43" src="https://github.com/user-attachments/assets/0e16fea9-b665-4770-aade-bab84dcc8783" />


# Have fun playing. :)



HINT: In case you get stuck and cannot think of any other words to progress, just try repeating goal word. ;)
