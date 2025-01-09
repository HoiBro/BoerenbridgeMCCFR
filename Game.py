import random
from copy import deepcopy
import numpy as np


class Game:
    """Class which keeps track of the rules of the game."""

    def __init__(self, deck, handsize):
        """Initialize a game for a given deck and handsize."""
        self.deck = deck
        self.handsize = handsize
        self.deck.reset_deck()
        self.mean = np.mean(deck.ranks)

    def suit_abstraction_dict(self, hand, trump):
        """Function to create a suit dictionary for card isomorphism.
        Each suit is mapped to an abstraction which is specific for the initial hand.
        "fourth" is the highest valued suit and "first" is the lowest."""
        suits = self.deck.suit
        suit_abstraction = ('first', 'second', 'third', 'fourth')
        suit_dict = {}
        check_val = set()

        hand.sort(key=lambda x: x[1])

        suit_dict[trump[0]] = suit_abstraction[len(suits) - 1]

        for card in hand:
            if card[0] not in check_val and card[0] != trump[0]:
                suit_dict[card[0]] = suit_abstraction[len(check_val)]
                check_val.add(card[0])
        for suit in suits:
            if suit not in check_val and suit != trump[0]:
                suit_dict[suit] = suit_abstraction[len(check_val)]
                check_val.add(suit)
        return suit_dict

    @staticmethod
    def translate_suits(game_state):
        """Function which translates all the suits in a game state, based on the suit dictionary."""
        suit_dict = game_state[5][game_state[0]]
        hand = game_state[1][game_state[0]]
        trump = game_state[1][2]
        hist = game_state[2]
        new_hand = []
        new_hist = ()
        for card in hand:
            new_card = (suit_dict[card[0]], card[1])
            new_hand.append(new_card)
        new_trump = (suit_dict[trump[0]], trump[1])
        for card in hist:
            if isinstance(card, np.int64):
                new_hist += (int(card),)
            else:
                new_hist += ((suit_dict[card[0]], card[1]),)
        return new_hand, new_trump, new_hist

    def sample_new_game(self, hands=None):
        """Function to sample an initial game state with an empty history. Optionality to provide hands instead of
        uniformly sampling a hand. Game state has form:
        (active_player, cards, history, player_wins, terminal, suit_dicts) where,
        active_player = 0 or 1 for player 0 or 1
        cards = a list of 2 lists and one trump, one list for each player, the lists contain cards which are tuples of the form: (suit, rank)
        history = tuple containing all actions in order
        player_wins = the amount of wins both players have gotten
        terminal = boolean indicating whether the state is terminal
        suit_dicts = list of suit dicts, one for each player"""
        if hands:
            cards = [sorted(hands[0]), sorted(hands[1]), hands[2]]
            suit_dicts = [self.suit_abstraction_dict(hands[0], hands[2]), self.suit_abstraction_dict(hands[1], hands[2])]
            return (0, cards, (), np.zeros(2, dtype=int), False, suit_dicts)
        else:
            game_cards = random.sample(self.deck.deck2, 2 * self.handsize + 1)
            cards = [sorted(game_cards[self.handsize + 1:]), sorted(game_cards[:self.handsize]), game_cards[self.handsize]]
            suit_dicts = [self.suit_abstraction_dict(cards[0], cards[2]), self.suit_abstraction_dict(cards[1], cards[2])]
            return (0, cards, (), np.zeros(2, dtype=int), False, suit_dicts)

    def get_possible_actions(self, game_state):
        """Function which uses a game state to determine the possible actions as a list from this game state."""
        cards = sorted(game_state[1][game_state[0]])

        # If betting
        if len(game_state[2]) < 2:
            return np.arange(self.handsize + 1)
        elif len(game_state[2]) % 2 == 0:
            return cards
        else:
            # Players have to play the same suit, unless they can not.
            possible = [card for card in cards if card[0] == game_state[2][-1][0]]
            if len(possible) == 0:
                return cards
            else:
                return possible

    def get_next_game_state(self, game_state, action):
        """Function which returns a new game state, based on the previous game state and the action taken."""
        terminal = False
        
        next_active_player = (game_state[0] + 1) % 2
        next_hands = deepcopy(game_state[1])
        history = game_state[2] + (action,)
        player_wins = deepcopy(game_state[3])

        if len(history) == 2 * self.handsize + 2:
            terminal = True

        if not isinstance(action, np.int64):
            next_hands[game_state[0]].remove(action)

        # The same player only plays twice in a row if they win a round as a reacting player.
        if len(history) > 2 and len(history) % 2 == 0 and (action[0] == game_state[2][-1][0] and game_state[2][-1][1] < action[1] or action[0] == game_state[1][2][0] and game_state[2][-1][0] != game_state[1][2][0]):
            next_active_player = game_state[0]
        
        if len(history) > 2 and len(history) % 2 == 0:
            player_wins[next_active_player] += 1

        return (next_active_player, next_hands, history, player_wins, terminal, game_state[5])
    
    def get_payoff(self, game_state):
        """Function to determine the payoff of a game for a given player."""
        p_bets = game_state[2][game_state[0]]
        p_wins = game_state[3][game_state[0]]
        o_bets = game_state[2][(game_state[0] + 1) % 2]
        o_wins = game_state[3][(game_state[0] + 1) % 2]

        if p_bets == p_wins and o_bets == o_wins:
            return p_wins - o_wins
        elif p_bets == p_wins:
            return 5 + p_wins + abs(o_bets - o_wins)
        elif o_bets == o_wins:
            return -5 - o_wins - abs(p_bets - p_wins)
        else:
            return abs(o_bets - o_wins) - abs(p_bets - p_wins)
