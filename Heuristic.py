from Infoset import Infoset
import random
import itertools
import numpy as np
import pickle
import os

class Heuristic:
    """Class which can generate a strategy, using heuristic rules."""

    def __init__(self, game, abstraction_function, std):
        """Initialize a game, a dictionary containing the strategy profile, an abstraction function and a standard deviation."""
        self.game = game
        self.infoset_dict = {}
        self.abstraction_function = abstraction_function
        self.infoset_data = (self.infoset_dict, self.abstraction_function)
        self.std = std
        self.max_chs = self.get_hand_strength(0)

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

    def reacting(self, game_state):
        """Returns false if first player of round and last played card in the history if not"""
        if len(game_state[2]) % 2 == 0:
            return False
        else:
            return game_state[2][-1]
    
    def get_hand_strength(self, player, game_state=None):
        """Get the compared hand strength of a player. If no game state is given returns the maximum hand strength"""
        deck = self.game.deck.deck2.copy()
        if game_state:
            hand = game_state[1][player]
            for card in hand:
                deck.remove(card)
            deck.remove(game_state[1][2])
        else:
            deck.sort(key= lambda x: x[1], reverse=True)
            hand = deck[:self.game.handsize]
            deck = deck[self.game.handsize + 1:]
        hand_strength = sum(map(lambda x: x[1], hand)) / len(hand)
        deck_strength = sum(map(lambda x: x[1], deck)) / len(deck)
        return hand_strength - deck_strength

    def heuristic(self, game_state):
        """Recursive function which updates the infosets. Use this function to define your heuristic strat"""

        # Base case
        if game_state[4]:
            return self.game.get_payoff(game_state)
        
        possible_actions = self.game.get_possible_actions(game_state)

        if len(possible_actions) == 1:
            self.heuristic(self.game.get_next_game_state(game_state, possible_actions[0]))
        else:
            info_key = self.get_info_key(game_state)
            infoset = self.get_infoset(info_key)
            react = self.reacting(game_state)

            # Define rules using the info key or game state
            # A passive player who plays highest card if winnable and lowest otherwise
            # the player bets according to their hand strength compared to the deck strength
            # with a normal distribution applied
            if len(game_state[2]) < 2:
                chs = self.get_hand_strength(game_state[0], game_state=game_state)
                chs = (((self.game.handsize + 1)*chs/self.max_chs)+self.game.handsize)/2
                index = round(random.normalvariate(chs, self.std))
                if index < 0:
                    index = 0
                elif index > self.game.handsize:
                    index = self.game.handsize
            elif react is not False and \
                    react[0] == possible_actions[0][0] and \
                    react[1] < max(possible_actions, key=lambda t: t[1])[1]:
                higher_list = [card for card in possible_actions if card[1] > react[1]]
                index = possible_actions.index(min(higher_list, key=lambda t: t[1]))
            elif react:
                # Play lowest card and randomize if more than one
                index = random.choice([i for i, x in enumerate(possible_actions) if
                                        x[1] == min(possible_actions, key=lambda t: t[1])[1]])
            else:
                index = random.choice([i for i, x in enumerate(possible_actions) if
                                        x[1] == max(possible_actions, key=lambda t: t[1])[1]])
            infoset.heuristic_update(index)

            for action in possible_actions:
                self.heuristic(self.game.get_next_game_state(game_state, action))
    
    def train_heuristic(self):
        """Fully train the heuristic algorithm by calling it at every chance node."""
        for trump in self.game.deck.ranks:
            self.game.deck.reset_deck()
            self.game.deck.deck1.remove((self.game.deck.suit[0], trump))
            for dealt_cards in itertools.combinations(self.game.deck.deck1, self.game.handsize * 2):
                for hand1 in itertools.combinations(dealt_cards, self.game.handsize):
                    hand2 = list(card for card in dealt_cards if card not in hand1)
                    hands = [sorted(list(hand1)), sorted(hand2), (self.game.deck.suit[0], trump)]
                    game_state = self.game.sample_new_game(hands=hands)
                    self.heuristic(game_state)

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
            self.dict_helper(self.game.get_next_game_state(game_state, action))

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
    
    def save_dict(self, name):
        """Save information dictionary as pickle."""
        filename = f"Dicts/{name}.pkl"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        a_file = open(filename, "wb")
        pickle.dump(self.infoset_dict, a_file)
        a_file.close()
    
    def load_dict(self, name):
        """Load information dictionary as pickle."""
        a_file = open(f"Dicts/{name}.pkl", "rb")
        output = pickle.load(a_file)
        self.infoset_dict = output
        self.infoset_data = (self.infoset_dict, self.abstraction_function)

    def play_round(self, first_player, verbose):
        """Recursive function for playing a round by sampling from the given infodictionary. And allowing the input to play.
        first_player: info_dicts index for starting player"""
        game_state = self.game.sample_new_game()
        if verbose:
            print(f"Your opponent's hand is: {game_state[1][(first_player + 1) % 2]}")
        print(f"Your hand is: {game_state[1][first_player]}")
        print(f"The trump is {game_state[1][2]}")
        print('')
        print(f"The history is {game_state[2]}")
        print('')
        while not game_state[4]:
            possible_actions = self.game.get_possible_actions(game_state)
            if game_state[0] == first_player:
                while True:
                    try:
                        if isinstance(possible_actions[0], np.int64):
                            index = int(input(f'Give the amount you would like to bet (maximimum of {self.game.handsize}): '))
                        else:
                            print(f"You have the following possible actions: {possible_actions}")
                            index = int(input('Give the index of the action you want to choose (starting from 0): '))
                    except ValueError:
                        print("Sorry, you didn't provide a valid index starting from 0, try again.")
                        continue
                    if index > len(possible_actions)-1:
                        print("Sorry, you didn't provide a valid index starting from 0, try again.")
                        continue
                    else:
                        break
                print('')
                action = possible_actions[index]
                game_state = self.game.get_next_game_state(game_state, action)
            else:
                if len(possible_actions) == 1:
                    if verbose:
                        print(f"Your opponent played the following action: {possible_actions[0]} as their only action")
                    else:
                        print(f"Your opponent played the following action: {possible_actions[0]}")

                    game_state = self.game.get_next_game_state(game_state, possible_actions[0])
                else:
                    info_key = self.get_info_key(game_state)
                    infoset = self.get_infoset(info_key)
                    strategy = infoset.get_average_strategy()
                    action = random.choices(possible_actions, strategy)[0]
                    if verbose:
                        print(f"Your opponent had the following possible actions: {possible_actions} with the "
                              f"following probabilities: {strategy}")
                    if isinstance(action, np.int64):
                        print(f"Your opponent bet {action}")
                    else:
                        print(f"Your opponent played the following action: {action}")
                    game_state = self.game.get_next_game_state(game_state, action)
                print('')

        # Determine the payoffs
        p_bets = game_state[2][first_player]
        p_wins = game_state[3][first_player]
        o_bets = game_state[2][(first_player + 1) % 2]
        o_wins = game_state[3][(first_player + 1) % 2]
        if p_bets == p_wins and o_bets == o_wins:
            return [10 + 2*p_bets, 10 + 2*o_bets]
        elif p_bets == p_wins:
            return [10 + 2*p_bets, -2*abs(o_bets - o_wins)]
        elif o_bets == o_wins:
            return [-2*abs(p_bets - p_wins), 10 + 2*o_bets]
        else:
            return [-2*abs(p_bets - p_wins), -2*abs(o_bets - o_wins)]

    def play_game(self, winning_score, verbose=False):
        """Play a first to n game as a player against the generated strategy."""
        score1 = 0
        score2 = 0
        i = 1
        print("Initializing new game...")
        print('')
        while score1 < winning_score and score2 < winning_score:
            i = (i + 1) % 2
            payoff = self.play_round(i, verbose)
            if payoff[0] > 0:
                print(f"You won with a score of {payoff[0]}")
            else:
                print(f"You lost with a score of {-1*payoff[0]}")
            if payoff[1] > 0:
                print(f"Your opponent won with a score of {payoff[1]}")
            else:
                print(f"Your opponent lost with a score of {-1*payoff[1]}")
            score1 += payoff[0]
            score2 += payoff[1]
            print('')
            print(f"The score is You: {score1}, Opponent: {score2}")
            print('')
            print('')
        final = 'won'
        if score2 > score1:
            final = 'lost'
        print(f"You {final} the match by {abs(score1 - score2)} points.")
        return score1, score2
