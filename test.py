from Deck import Deck
from Game import Game
from MCCFR import MCCFR
from Abstraction_functions import simple, identity, naive, possible, advanced

suits = 2
ranks = 6
hand_size = 4

deck = Deck(suits, ranks)
game = Game(deck, hand_size)
mccfr = MCCFR(game, identity)

mccfr.play_game(15, True)


print(deck.deck1)