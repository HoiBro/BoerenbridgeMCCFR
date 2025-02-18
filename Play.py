from Infoset import Infoset
import random


class Play:
    """Class to simulate games of Boerenbridge."""

    def __init__(self, game, infoset_data_1, infoset_data_2):
        """Initialize a game, two dictionary containing the strategy profiles and abstraction functions."""
        self.game = game
        self.infoset_dicts = [infoset_data_1[0].copy(), infoset_data_2[0].copy()]
        self.abstraction_functions = [infoset_data_1[1], infoset_data_2[1]]

    def get_infoset(self, info_key, player):
        """Create an infoset if needed and return."""
        if info_key not in self.infoset_dicts[player]:
            self.infoset_dicts[player][info_key] = Infoset(info_key)
        return self.infoset_dicts[player][info_key]

    def get_info_key(self, game_state, player):
        """Function which generates an info_key, given a game_state. First the suits are abstracted using the
        suit dict, after which the abstraction function is used for further abstraction."""
        possible_action = self.game.get_possible_actions(game_state)
        possible_action_len = len(possible_action)
        new_hand, new_trump, new_hist = self.game.translate_suits(game_state)
        abs_hand, abs_trump, abs_hist, abs_wins = self.abstraction_functions[player](new_hand, new_trump, new_hist, game_state[3], possible_action, self.game.mean)
        key = (game_state[0], frozenset(abs_hand), abs_trump, abs_hist, abs_wins[0], abs_wins[1], possible_action_len)
        return key

    def play_round(self, first_player):
        """Recursive function for playing a round by sampling from the given infodicts.
        first_player: info_dicts index for starting player"""
        game_state = self.game.sample_new_game()
        while not game_state[4]:
            possible_actions = self.game.get_possible_actions(game_state)
            if len(possible_actions) == 1:
                game_state = self.game.get_next_game_state(game_state, possible_actions[0])
            else:
                infoset_dict_index = game_state[0]
                if first_player == 1:
                    infoset_dict_index = (infoset_dict_index + 1) % 2
                info_key = self.get_info_key(game_state, infoset_dict_index)
                infoset = self.get_infoset(info_key, infoset_dict_index)
                strategy = infoset.get_average_strategy()
                action = random.choices(possible_actions, strategy)[0]
                game_state = self.game.get_next_game_state(game_state, action)

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

    def play_n_rounds(self, iterations):
        """Play iterations, number of rounds and return the first players average and total score."""
        player_payoff = 0
        total_payoff = 0
        for _ in range(iterations):
            for i in range(2):
                payoff = self.play_round(i)
                player_payoff += payoff[0] * ((i * -2) + 1)
                total_payoff += (payoff[0] - payoff[1]) * ((i * -2) + 1)
        return [total_payoff / (iterations * 2), player_payoff]
