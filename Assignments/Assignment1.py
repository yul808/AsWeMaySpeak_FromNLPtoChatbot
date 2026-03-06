# News Headlines
import tracery
from tracery.modifiers import base_english

rules = {
    "origin":"NEWS: #name# #verb# to #o# #country#.",
    "name": ["Donald Trump","Gandhi","Voldemort","Spiderman", "LeBron James"],
    "o": ["make a deal with", "militarily free", "flee to", "take over", "steal from"],
    "verb": ["wants", "has", "plans", "claims", "prepares", "stops tariffs"],
    "country": ["Afghanistan", "Russia", "the USA", "China", "Pandora", "Mexico", "Bayern", "Switzerland", "Argentina" ]
}
# modifiers: .a, .s, .capitalize
grammar = tracery.Grammar(rules)
grammar.add_modifiers(base_english)
for i in range(10):
    print(grammar.flatten("#origin#"))