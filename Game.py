import random
from copy import deepcopy
import numpy as np


class Game:
    """Class which keeps track of the rules of the game."""

    def __init__(self, deck, handsize):
        """Initialize a game for a given deck, handsize and betting type."""
        self.deck = deck
        self.handsize = handsize
        self.deck.reset_deck()
        self.bets = []
        self.wins = []
        self.mean = np.mean(deck.ranks)

    def suit_abstraction_dict(self, hand, identity=False):
        """Function to create a suit dict for card isomorphism. Each suit is mapped to an abstraction which
        is specific for the initial hand."""
        suits = self.deck.suit
        suit_abstraction = ('first', 'second', 'third', 'fourth')
        suit_dict = {}
        check_val = set()

        hand.sort(key=lambda x: x[1])

        for card in hand:
            if card[0] not in check_val:
                suit_dict[card[0]] = suit_abstraction[len(check_val)]
                check_val.add(card[0])
        for suit in suits:
            if suit not in check_val:
                suit_dict[suit] = suit_abstraction[len(check_val)]
                check_val.add(suit)
        if identity:
            suit_dict = {'clubs': 'first', 'diamonds': 'second', 'hearts': 'third', 'spades': 'fourth'}
        return suit_dict

    @staticmethod
    def translate_suits(game_state):
        """Function which translates all the suits in a game state, based on the suit dict."""
        suit_dict = game_state[4][game_state[0]]
        hand = game_state[1][game_state[0]]
        hist = game_state[2]
        new_hand = []
        new_hist = ()
        for card in hand:
            new_card = (suit_dict[card[0]], card[1])
            new_hand.append(new_card)
        for card in hist:
            if len(card) == 2:
                new_hist += ((suit_dict[card[0]], card[1]),)
            else:
                new_hist += (card,)
        return new_hand, new_hist

    def sample_new_game(self, hands=None):
        """Function to sample an initial game state with an empty history. Optionality to provide hands instead of
        uniformly sampling a hand. Game state has form:
        (active_player, hands, history, status, suit_dicts) where,
        active_player = 0 or 1 for player 0 or 1.
        hands = a list of lists, one for each player, containing cards which are tuples of the form: (suit, rank)
        history = tuple containing all actions in order
        status = integer representing the status of the game, 0 for card playing actions, 1 for betting and 2 for a terminal node
        suit_dicts = list of suit_dicts, one for each player"""
        if hands:
            hands = [sorted(hands[0]), sorted(hands[1])]
            suit_dicts = [self.suit_abstraction_dict(hands[0]), self.suit_abstraction_dict(hands[1])]
            return (0, hands, (), 1, suit_dicts)
        else:
            two_hands = random.sample(self.deck.deck2, 2 * self.handsize)
            hands = [sorted(two_hands[self.handsize:]), sorted(two_hands[:self.handsize])]
            suit_dicts = [self.suit_abstraction_dict(hands[0]), self.suit_abstraction_dict(hands[1])]
            return (0, hands, (), 1, suit_dicts)

    def get_possible_actions(self, game_state):
        """Function which uses a game state to determine the possible actions as a list from this game state."""
        cards = sorted(game_state[1][game_state[0]])

        if game_state[3] == 1:
            return np.arange(self.handsize + 1)
        elif len(game_state[2]) % 2 == 0:
            return cards
        else:
            # Players have to play the same suit, unless they can not.
            possible = [i for i in cards if i[0] == game_state[2][-1][0]]
            if len(possible) == 0:
                return cards
            else:
                return possible

    def get_next_game_state(self, game_state, action):
        """Function which returns a new game state, based on the previous game state and the action taken."""
        game_status = 0

        if len(game_state[2]) < 2:
            game_status = 1
        elif len(game_state[2]) == (2 * self.handsize) - 1:
            game_status = 2

        next_active_player = (game_state[0] + 1) % 2
        next_hands = deepcopy(game_state[1])

        # The same player only plays twice in a row if she wins a round as a reacting player.
        if (len(game_state[2]) % 2 == 1) & ((action[0] == game_state[2][-1][0]) & (game_state[2][-1][1] < action[1])):
            next_active_player = game_state[0]

        if len(game_state[2]) > 1:
            history = game_state[2] + (action,)
            next_hands[game_state[0]].remove(action)
        else:
            self.bets[game_state[0]] += action

        return (next_active_player, next_hands, history, game_status, game_state[4])
    
    def get_payoff(self, game_state):
        """Function to determine the payoff of a game for a given player."""
        player = game_state[0]
        opponent = (player + 1) % 2

        if self.bets[player] == self.wins[player] & self.bets[opponent] == self.wins[opponent]:
            return 2*(self.wins[player] - self.wins[opponent])
        elif self.bets[player] == self.wins[player]:
            return 10 + 2*(self.wins[player] + abs(self.bets[opponent] - self.wins[opponent]))
        elif self.bets[opponent] == self.wins[opponent]:
            return -10 - 2*(self.wins[opponent] + abs(self.bets[player] - self.wins[player]))
        else:
            return 2*(abs(self.bets[opponent] - self.wins[opponent]) - abs(self.bets[player] - self.wins[player]))
