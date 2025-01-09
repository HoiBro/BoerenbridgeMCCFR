from Infoset import Infoset
import random
import itertools
import numpy as np

class Heuristic:
    """Class which can generate a strategy, using heuristic rules."""

    def __init__(self, game, abstraction_function, confidence):
        """Initialize a game, a dictionary containing the strategy profile and an abstraction function and confidence."""
        self.game = game
        self.infoset_dict = {}
        self.abstraction_function = abstraction_function
        self.infoset_data = (self.infoset_dict, self.abstraction_function)
        self.confidence = confidence

    def get_infoset(self, info_key):
        """Create an infoset if needed and return."""
        if info_key not in self.infoset_dict:
            self.infoset_dict[info_key] = Infoset(info_key)
        return self.infoset_dict[info_key]

    def get_info_key(self, game_state):
        """Function which generates an info_key, given a game_state. First the suits are abstracted using the
        suit dict, after which the abstraction function is used for further abstraction."""
        possible_action = self.game.get_possible_actions(game_state)
        possible_action_len = len(possible_action)
        new_hand, new_trump, new_hist = self.game.translate_suits(game_state)
        abs_hand, abs_trump, abs_hist, abs_wins = self.abstraction_function(new_hand, new_trump, new_hist, game_state[3], possible_action, self.game.mean)
        key = (game_state[0], frozenset(abs_hand), abs_trump, abs_hist, abs_wins[0], abs_wins[1], possible_action_len)
        return key

    def avg_deck(self, game_state):
        """Function which finds the average rank of the deck, without the cards that have been played"""
        deck = self.game.deck.deck2.copy()
        for action in game_state[2]:
            if not isinstance(action, np.int64):
                deck.remove(action)
        return sum(map(lambda x: x[1], deck)) / len(deck)

    def reacting(self, game_state):
        """return False if first player of round and first card if not"""
        if len(game_state[2]) % 2 == 0:
            return False
        else:
            return game_state[2][-1]

    def heuristic(self, game_state):
        """Recursive function which updates the infosets. Use this function to define your heuristic strat"""
        if game_state[4]:
            return self.game.get_payoff(game_state)
        possible_actions = self.game.get_possible_actions(game_state)

        if len(possible_actions) == 1:
            self.heuristic(self.game.get_next_game_state(game_state, possible_actions[0]))
        else:
            info_key = self.get_info_key(game_state)
            infoset = self.get_infoset(info_key)

            # Define rules using the info_key
            # A passive player who plays highest card if winnable and lowest otherwise
            if self.reacting(game_state) is not False and \
                    self.reacting(game_state)[0] == possible_actions[0][0] and \
                    self.reacting(game_state)[1] < max(possible_actions, key=lambda t: t[1])[1]:
                higher_list = [i for i in possible_actions if i[1] > self.reacting(game_state)[1]]
                index = possible_actions.index(min(higher_list, key=lambda t: t[1]))

            elif self.reacting(game_state):
                # Play lowest card and randomize if more than one
                index = random.choice([i for i, x in enumerate(possible_actions) if
                                        x[1] == min(possible_actions, key=lambda t: t[1])[1]])
            else:
                index = random.choice([i for i, x in enumerate(possible_actions) if
                                        x[1] == max(possible_actions, key=lambda t: t[1])[1]])
            infoset.heuristic_update(index)

            for action in possible_actions:
                self.heuristic(self.game.get_next_game_state(game_state, action))

    def count_infosets(self):
        """Count total number of infosets in the infodictionary."""
        p1_count = len([x for x, _ in self.infoset_dict.items() if x[0] == 0])
        p2_count = len(self.infoset_dict.items()) - p1_count
        return p1_count, p2_count

    def dict_helper(self, game_state):
        """Function which creates an infodictionary for every branching game state from an original game state."""
        if game_state[4]:
            return
        possible_actions = self.game.get_possible_actions(game_state)

        if len(possible_actions) != 1:
            info_key = self.get_info_key(game_state)
            self.get_infoset(info_key)

        for action in possible_actions:
            next_game_state = self.game.get_next_game_state(game_state, action)
            self.dict_helper(next_game_state)

    def make_dict(self):
        """Create the infodictionary for all possible infosets."""
        for trump in self.game.deck.ranks:
            self.game.deck.reset_deck()
            self.game.deck.deck1.remove((self.game.deck.suit[0], trump))
            for dealt_cards in itertools.combinations(self.game.deck.deck1, self.game.handsize * 2):
                for hand1 in itertools.combinations(dealt_cards, self.game.handsize):
                    hand2 = list(card for card in dealt_cards if card not in hand1)
                    hands = [sorted(list(hand1)), sorted(hand2), (self.game.deck.suit[0], trump)]
                    game_state = self.game.sample_new_game(hands=hands)
                    self.dict_helper(game_state)