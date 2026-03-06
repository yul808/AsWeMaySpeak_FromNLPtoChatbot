# OPTION 1
# 1. Find a text corpus that interests you
# 2. Use spaCy to process and harvest groups of words/phrases
# 3. Then use `tracery`to build a text generator based on the harvested material

import tracery
from tracery.modifiers import base_english

import spacy
nlp = spacy.load("en_core_web_md")

# Chapters 1-3 from Fyodor Dostoyevsky's "Crime And Punishment"
text = open("CrimeAndPunishment").read()
doc = nlp(text)

all_words = [t for t in doc if t.is_alpha]

NOUNs = [t.lemma_ for t in all_words if t.pos_ == "NOUN"]
VERBs = [t.lemma_ for t in all_words if t.pos_ == "VERB"]
ADJs  = [t.lemma_ for t in all_words if t.pos_ == "ADJ"]
ADVs  = [t.lemma_ for t in all_words if t.pos_ == "ADV"]
PRONs = [t.lemma_ for t in all_words if t.pos_ == "PRON"]
ADPs  = [t.lemma_ for t in all_words if t.pos_ == "ADP"]
DETs  = [t.lemma_ for t in all_words if t.pos_ == "DET"]
AUXs  = [t.lemma_ for t in all_words if t.pos_ == "AUX"]

# optional: fallback values if the text lacks one category (suggestion by ChatGPT)
if not DETs: DETs = ["the"]
if not PRONs: PRONs = ["they"]
if not AUXs: AUXs = ["is"]

# Tracery grammar using harvested lists
rules = {
    "origin": [
        "#sentence_1#",
        "#sentence_2#",
        "#sentence_3#",
        "#sentence_4#",
        "#sentence_5#"
    ],

    "sentence_1": "#Det.capitalize# #Noun.capitalize# #Verb# #Det# #Noun#.",
    "sentence_2": "#Det.capitalize# #Noun.capitalize# #Verb# #Prep# #Det# #Noun#.",
    "sentence_3": "#Pronoun.capitalize# #Aux# #Adverb# #Verb# #Det# #Noun#.",
    "sentence_4": "#Det.capitalize# #Noun.capitalize# #Verb# #Det# #Noun# #Prep# #Det# #Noun#.",
    "sentence_5": "#Pronoun.capitalize# #Aux# #Det# #Adjective# #Noun# #Verb#.",

    "Noun": NOUNs,
    "Verb": VERBs,
    "Adjective": ADJs,
    "Adverb": ADVs,
    "Pronoun": PRONs,
    "Prep": ADPs,
    "Det": DETs,
    "Aux": AUXs
}

grammar = tracery.Grammar(rules)
grammar.add_modifiers(base_english)

# Generating examples
for i in range(10):
    print(grammar.flatten("#origin#"))
