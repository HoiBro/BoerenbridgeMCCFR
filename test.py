from Deck import Deck
from Game import Game
from MCCFR import MCCFR
from Abstraction_functions import identity, simple, naive, advanced
from Infoset import Infoset

suits = 2
ranks = 3
hand_size = 2

deck = Deck(suits, ranks)
game = Game(deck, hand_size)
mccfr = MCCFR(game, identity)

# new_game = game.sample_new_game()
#test = mccfr.evaluate()
#print(test)
mccfr.load_dict("SmolTest1")
print(mccfr.count_infosets())
# infoset_dict_keys = list(mccfr.infoset_dict.keys())
# infoset_dict_keys.sort()
# sd = {i: mccfr.infoset_dict[i] for i in infoset_dict_keys}
# print(sd)
# print(mccfr.infoset_dict)
# print(f"{mccfr.infoset_dict[(1, frozenset({('first', 13)}), ('second', 12), (1,), 2)].regret_matching()}")
mccfr.play_game(100)

# print(new_game[1])
# print(deck.deck1)
# print(game.translate_suits(new_game))