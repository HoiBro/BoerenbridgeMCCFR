import math
import numpy as np
from tqdm import tqdm
import random
from copy import deepcopy
import itertools
from Infoset import Infoset
import pickle
import os


class MCCFR:
    """Class to run the MCCFR algorithm."""

    def __init__(self, game, abstraction_function):
        """Initialize a game, a dictionary containing the strategy profile and an abstraction function."""
        self.game = game
        self.infoset_dict = {}
        self.abstraction_function = abstraction_function
        self.infoset_data = (self.infoset_dict, self.abstraction_function)

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

    def chance_cfr(self, game_state, reach_probs):
        """Recursive function for chance sampled MCCFR."""

        # Base case
        if game_state[4]:
            return self.game.get_payoff(game_state)

        possible_actions = self.game.get_possible_actions(game_state)
        counterfactual_values = np.zeros(len(possible_actions))
        return_value = -1

        # If only 1 possible action, no strategy required.
        if len(possible_actions) == 1:
            next_game_state = self.game.get_next_game_state(game_state, possible_actions[0])
            if game_state[0] == next_game_state[0]:
                return_value = 1
            node_value = return_value * self.chance_cfr(next_game_state, reach_probs)

        else:
            player = game_state[0]
            opponent = (player + 1) % 2
            info_key = self.get_info_key(game_state)
            infoset = self.get_infoset(info_key)
            strategy = infoset.regret_matching()
            infoset.update_strategy_sum(reach_probs[player])

            for ix, action in enumerate(possible_actions):
                action_prob = strategy[ix]

                # Compute new reach probabilities after this action
                new_reach_probs = reach_probs.copy()
                new_reach_probs[player] *= action_prob

                # Recursively call MCCFR
                next_game_state = self.game.get_next_game_state(game_state, action)
                if game_state[0] == next_game_state[0]:
                    return_value = 1
                counterfactual_values[ix] = return_value * self.chance_cfr(next_game_state, new_reach_probs)

            # Value of the current game state is counterfactual values weighted by the strategy
            node_value = counterfactual_values.dot(strategy)

            for ix, action in enumerate(possible_actions):
                infoset.cumulative_regrets[ix] += reach_probs[opponent] * (counterfactual_values[ix] - node_value)
        return node_value

    def external_cfr(self, game_state, reach_probs, update_player):
        """Recursive function for external sampled MCCFR."""

        # Base case
        if game_state[4]:
            return self.game.get_payoff(game_state)

        possible_actions = self.game.get_possible_actions(game_state)
        counterfactual_values = np.zeros(len(possible_actions))
        return_value = -1

        # If only 1 possible action, no strategy required.
        if len(possible_actions) == 1:
            next_game_state = self.game.get_next_game_state(game_state, possible_actions[0])
            if game_state[0] == next_game_state[0]:
                return_value = 1
            node_value = return_value * self.external_cfr(next_game_state, reach_probs, update_player)

        else:
            player = game_state[0]
            opponent = (game_state[0] + 1) % 2
            info_key = self.get_info_key(game_state)
            infoset = self.get_infoset(info_key)
            strategy = infoset.regret_matching()

            # External gets sampled
            if player != update_player:
                action = random.choices(possible_actions, strategy)[0]
                action_index = list(possible_actions).index(action)
                action_prob = strategy[action_index]

                # Compute new reach probabilities after this action
                new_reach_probs = reach_probs.copy()
                new_reach_probs[player] *= action_prob

                next_game_state = self.game.get_next_game_state(game_state, action)
                if game_state[0] == next_game_state[0]:
                    return_value = 1
                node_value = return_value * self.external_cfr(next_game_state, new_reach_probs, update_player)

            else:
                infoset.update_strategy_sum(reach_probs[player])
                for ix, action in enumerate(possible_actions):
                    action_prob = strategy[ix]

                    # Compute new reach probabilities after this action
                    new_reach_probs = reach_probs.copy()
                    new_reach_probs[player] *= action_prob

                    # Recursively call MCCFR
                    next_game_state = self.game.get_next_game_state(game_state, action)
                    if game_state[0] == next_game_state[0]:
                        return_value = 1
                    counterfactual_values[ix] = return_value * self.external_cfr(next_game_state, new_reach_probs, update_player)

                # Value of the current game state is counterfactual values weighted by the strategy
                node_value = counterfactual_values.dot(strategy)

                for ix, action in enumerate(possible_actions):
                    infoset.cumulative_regrets[ix] += reach_probs[opponent] * (counterfactual_values[ix] - node_value)
        return node_value
    
    def train_cfr(self, num_iterations):
        """Train vanilla cfr by calling chance mccfr at every chance node."""
        util = 0
        games = 0
        for _ in range(num_iterations):
            for trump in self.game.deck.ranks:
                self.game.deck.reset_deck()
                self.game.deck.deck1.remove((self.game.deck.suit[0], trump))
                for dealt_cards in itertools.combinations(self.game.deck.deck1, self.game.handsize * 2):
                    for hand1 in itertools.combinations(dealt_cards, self.game.handsize):
                        hand2 = list(card for card in dealt_cards if card not in hand1)
                    
                        hands = [sorted(list(hand1)), sorted(hand2), (self.game.deck.suit[0], trump)]
                        game_state = self.game.sample_new_game(hands=hands)

                        reach_prob = np.ones(2)
                        games += 1
                        util += self.chance_cfr(game_state, reach_prob)
        return util / num_iterations * games

    def train_chance(self, num_iterations):
        """Train chance mccfr by calling the recursive function, iteration number of times."""
        util = 0
        for _ in range(num_iterations):
            game_state = self.game.sample_new_game()
            reach_probs = np.ones(2)
            util += self.chance_cfr(game_state, reach_probs)
        return util / num_iterations

    def train_external(self, num_iterations):
        """Train external mccfr by calling the recursive function, iteration number of times."""
        util = 0
        for _ in tqdm(range(num_iterations)):
            for i in range(2):
                game_state = self.game.sample_new_game()
                reach_probs = np.ones(2)
                util += self.external_cfr(game_state, reach_probs, i)
        return util / (num_iterations * 2)

    def count_infosets(self):
        """Function which counts the number of information sets in the infodictionary"""
        p1_count = len([x for x, _ in self.infoset_dict.items() if x[0] == 0])
        p2_count = len(self.infoset_dict.items()) - p1_count
        return p1_count, p2_count

    def evaluate_helper(self, game_state, reach_prob):
        """Function which recursively finds the expected utility."""

        # Base case
        if game_state[4]:
            return self.game.get_payoff(game_state)

        possible_actions = self.game.get_possible_actions(game_state)
        partial_values = np.zeros(len(possible_actions))
        return_value = -1

        # If only 1 possible action, no strategy required.
        if len(possible_actions) == 1:
            next_game_state = self.game.get_next_game_state(game_state, possible_actions[0])
            if game_state[0] == next_game_state[0]:
                return_value = 1
            node_value = return_value * self.evaluate_helper(next_game_state, reach_prob)

        else:
            info_key = self.get_info_key(game_state)
            infoset = self.get_infoset(info_key)
            strategy = infoset.get_average_strategy()

            for ix, action in enumerate(possible_actions):
                action_prob = strategy[ix]

                # Compute new reach probabilities after this action
                new_reach_prob = reach_prob
                new_reach_prob *= action_prob

                # Recursively call evaluate function
                next_game_state = self.game.get_next_game_state(game_state, action)
                if game_state[0] == next_game_state[0]:
                    return_value = 1
                partial_values[ix] = return_value * self.evaluate_helper(next_game_state, new_reach_prob)

            # Value of the current game state is counterfactual values weighted by the strategy
            node_value = partial_values.dot(strategy)
        return node_value

    def evaluate(self):
        """Evaluates the current infodictionary by multiplying the probabilities of the
        terminal nodes with the utilities of those nodes"""
        hand_prob = (len(self.game.deck.deck2) *
                     math.comb(len(self.game.deck.deck2) - 1, self.game.handsize) *
                     math.comb(len(self.game.deck.deck2) - self.game.handsize - 1, self.game.handsize))
        util = 0
        for trump in self.game.deck.ranks:
            self.game.deck.reset_deck()
            self.game.deck.deck1.remove((self.game.deck.suit[0], trump))
            for dealt_cards in itertools.combinations(self.game.deck.deck1, self.game.handsize * 2):
                for hand1 in itertools.combinations(dealt_cards, self.game.handsize):
                    hand2 = list(card for card in dealt_cards if card not in hand1)
                    
                    hands = [sorted(list(hand1)), sorted(hand2), (self.game.deck.suit[0], trump)]
                    game_state = self.game.sample_new_game(hands=hands)

                    reach_prob = 1
                    util += self.evaluate_helper(game_state, reach_prob)
        return len(self.game.deck.suit) * util / hand_prob

    def get_exploitability(self, num_iterations):
        """Use MCCFR to update just one player and evaluate after. Doing this for both players approximates the
        exploitability."""
        info_dict_copy = deepcopy(self.infoset_dict)
        for _ in range(num_iterations):
            game_state = self.game.sample_new_game()
            reach_probs = np.ones(2)
            self.external_cfr(game_state, reach_probs, 0)
        b_1 = self.evaluate()
        self.infoset_dict = deepcopy(info_dict_copy)
        for _ in range(num_iterations):
            game_state = self.game.sample_new_game()
            reach_probs = np.ones(2)
            self.external_cfr(game_state, reach_probs, 1)
        b_2 = self.evaluate()
        self.infoset_dict = deepcopy(info_dict_copy)
        return b_1 - b_2
    
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
    
    def complexity_pos(self):
        """Find the upper bound of the number of possible histories,
        this does not yet include the possibility for player cards having matching suits."""
        suits = len(self.game.deck.suit)
        ranks = len(self.game.deck.ranks)
        complexity = math.comb(suits*ranks, 2*self.game.handsize)*math.comb(2*self.game.handsize, self.game.handsize)*(suits*ranks - 2*self.game.handsize)
        complexity *= (self.game.handsize + 1)**2
        for i in range(self.game.handsize):
            complexity *= (i+1)**2
        return complexity

    def complexity_info(self):
        """Find the upper bound of the number of possible information sets,
        this does not yet include the possibility for player cards having matching suits."""
        suits = len(self.game.deck.suit)
        ranks = len(self.game.deck.ranks)
        complexity = math.comb(suits*ranks, self.game.handsize)*(suits*ranks - self.game.handsize)
        complexity *= (self.game.handsize + 1)**2
        for i in range(1, self.game.handsize):
            complexity *= (suits*ranks) - (self.game.handsize+i)
            complexity *= (i+1)
        return complexity

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
