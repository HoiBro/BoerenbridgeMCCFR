from Deck import Deck
from Game import Game
from MCCFR import MCCFR
from Abstraction_functions import simple, identity, naive, possible, advanced

suits = 4
ranks = 13
hand_size = 4

deck = Deck(suits, ranks)
game = Game(deck, hand_size)
mccfr = MCCFR(game, identity)

# new_game = game.sample_new_game()
mccfr.play_game(15, True)


# print(deck.deck1)
# print(game.translate_suits(new_game))