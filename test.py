from Deck import Deck
from Game import Game
from MCCFR import MCCFR
from Abstraction_functions import identity, simple, naive, advanced

suits = 2
ranks = 3
hand_size = 2

deck = Deck(suits, ranks)
game = Game(deck, hand_size)
mccfr = MCCFR(game, identity)

# new_game = game.sample_new_game()
mccfr.load_dict("test1")
mccfr.play_game(100, True)

# print(new_game[1])
# print(deck.deck1)
# print(game.translate_suits(new_game))