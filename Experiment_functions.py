from tqdm import tqdm
from Deck import Deck
from Game import Game
from MCCFR import MCCFR
from Abstraction_functions import identity, simple, simple_hand, naive, bets, suit, suitbet, advanced
from Play import Play
import wandb

#Just as a warning, this code is an absolute mess and I could've probably done this WAY better using string manipulation and a for loop but ah well, it is what it is now (I'm sorry :/).

def exploit(suits, ranks, hand_size, starting_iterations, train_iterations, intervals, eval_iterations, name):
    """Function to run the MCCFR experiment."""
    iterations_per_interval = int(train_iterations / intervals)
    deck = Deck(suits, ranks)
    game = Game(deck, hand_size)
    mccfr = MCCFR(game, identity)
    if starting_iterations != 0:
        mccfr.train_external(starting_iterations)
        wandb.log({f"exploitability": mccfr.get_exploitability(eval_iterations),
                    'iteration': starting_iterations})

    for i in tqdm(range(intervals), leave=False):
        mccfr.train_external(iterations_per_interval)
        wandb.log({f"exploitability": mccfr.get_exploitability(eval_iterations),
                   'iteration': starting_iterations + ((i+1)*iterations_per_interval)})
    if name != '':
        mccfr.save_dict(name)


def fast(suits, ranks, hand_size, train_iterations, name):
    """Function to generate an infodictionary."""
    deck = Deck(suits, ranks)
    game = Game(deck, hand_size)
    mccfr = MCCFR(game, identity)
    for _ in tqdm([1]):
        mccfr.train_external(train_iterations)
    mccfr.save_dict(name)


def fast_abs(suits, ranks, hand_size, train_iterations, name, abstraction):
    """Function to generate an infodictionary for a given abstraction."""
    deck = Deck(suits, ranks)
    game = Game(deck, hand_size)
    mccfr_abs = MCCFR(game, abstraction)
    for _ in tqdm([1]):
        mccfr_abs.train_external(train_iterations)
    mccfr_abs.save_dict(name)


def full_abstraction(suits, ranks, hand_size, starting_iterations, train_iterations, intervals, eval_iterations, name, abstractions):
    """Function to run the abstraction experiment using multiple abstraction methods."""
    iterations_per_interval = int(train_iterations / intervals)
    deck = Deck(suits, ranks)
    game = Game(deck, hand_size)
    mccfr = MCCFR(game, identity)

    if starting_iterations == 0:
        infoset_size_normal = sum(mccfr.count_infosets())
        wandb.log({"infoset_size_normal": infoset_size_normal, 'iteration': 0})

        if abstractions[0]:
            mccfr_abs_naive = MCCFR(game, naive)
            infoset_size_naive = sum(mccfr_abs_naive.count_infosets())
            play_naive = Play(game, mccfr_abs_naive.infoset_data, mccfr.infoset_data)
            result_naive = play_naive.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_naive": infoset_size_naive, "points_naive": result_naive[1], "result_naive": result_adv[0]})

        if abstractions[1]:
            mccfr_abs_sim = MCCFR(game, simple)
            infoset_size_sim = sum(mccfr_abs_sim.count_infosets())
            play_sim = Play(game, mccfr_abs_sim.infoset_data, mccfr.infoset_data)
            result_sim = play_sim.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_simple": infoset_size_sim, "points_simple": result_sim[1], "result_simple": result_sim[0]})

        if abstractions[2]:
            mccfr_abs_simh = MCCFR(game, simple_hand)
            infoset_size_simh = sum(mccfr_abs_simh.count_infosets())
            play_simh = Play(game, mccfr_abs_simh.infoset_data, mccfr.infoset_data)
            result_simh = play_simh.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_simplehand": infoset_size_simh, "points_simplehand": result_simh[1], "result_simplehand": result_simh[0]})

        if abstractions[3]:
            mccfr_abs_bets = MCCFR(game, bets)
            infoset_size_bets = sum(mccfr_abs_bets.count_infosets())
            play_bets = Play(game, mccfr_abs_bets.infoset_data, mccfr.infoset_data)
            result_bets = play_bets.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_bets": infoset_size_bets, "points_bets": result_bets[1], "result_bets": result_bets[0]})

        if abstractions[4]:
            mccfr_abs_suit = MCCFR(game, suit)
            infoset_size_suit = sum(mccfr_abs_suit.count_infosets())
            play_suit = Play(game, mccfr_abs_suit.infoset_data, mccfr.infoset_data)
            result_suit = play_suit.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_suit": infoset_size_suit, "points_suit": result_suit[1], "result_suit": result_suit[0]})

        if abstractions[5]:
            mccfr_abs_suitbet = MCCFR(game, suitbet)
            infoset_size_suitbet = sum(mccfr_abs_suitbet.count_infosets())
            play_suitbet = Play(game, mccfr_abs_suitbet.infoset_data, mccfr.infoset_data)
            result_suitbet = play_suitbet.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_suitbet": infoset_size_suitbet, "points_suitbet": result_suitbet[1], "result_suitbet": result_suitbet[0]})

        if abstractions[6]:
            mccfr_abs_adv = MCCFR(game, advanced)
            infoset_size_adv = sum(mccfr_abs_adv.count_infosets())
            play_adv = Play(game, mccfr_abs_adv.infoset_data, mccfr.infoset_data)
            result_adv = play_adv.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_adv": infoset_size_adv, "points_adv": result_adv[1], "result_adv": result_adv[0]})
    else:
        mccfr.train_external(starting_iterations)
        infoset_size_normal = sum(mccfr.count_infosets())
        wandb.log({"infoset_size_normal": infoset_size_normal, 'iteration': starting_iterations})

        if abstractions[0]:
            mccfr_abs_naive = MCCFR(game, naive)
            mccfr_abs_naive.train_external(starting_iterations)
            infoset_size_naive = sum(mccfr_abs_naive.count_infosets())
            play_naive = Play(game, mccfr_abs_naive.infoset_data, mccfr.infoset_data)
            result_naive = play_naive.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_naive": infoset_size_naive, "points_naive": result_naive[1], "result_naive": result_adv[0]})
        
        if abstractions[1]:
            mccfr_abs_sim = MCCFR(game, simple)
            mccfr_abs_sim.train_external(starting_iterations)
            infoset_size_sim = sum(mccfr_abs_sim.count_infosets())
            play_sim = Play(game, mccfr_abs_sim.infoset_data, mccfr.infoset_data)
            result_sim = play_sim.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_simple": infoset_size_sim, "points_simple": result_sim[1], "result_simple": result_sim[0]})

        if abstractions[2]:
            mccfr_abs_simh = MCCFR(game, simple_hand)
            mccfr_abs_simh.train_external(starting_iterations)
            infoset_size_simh = sum(mccfr_abs_simh.count_infosets())
            play_simh = Play(game, mccfr_abs_simh.infoset_data, mccfr.infoset_data)
            result_simh = play_simh.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_simplehand": infoset_size_simh, "points_simplehand": result_simh[1], "result_simplehand": result_simh[0]})

        if abstractions[3]:
            mccfr_abs_bets = MCCFR(game, bets)
            mccfr_abs_bets.train_external(starting_iterations)
            infoset_size_bets = sum(mccfr_abs_bets.count_infosets())
            play_bets = Play(game, mccfr_abs_bets.infoset_data, mccfr.infoset_data)
            result_bets = play_bets.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_bets": infoset_size_bets, "points_bets": result_bets[1], "result_bets": result_bets[0]})

        if abstractions[4]:
            mccfr_abs_suit = MCCFR(game, suit)
            mccfr_abs_suit.train_external(starting_iterations)
            infoset_size_suit = sum(mccfr_abs_suit.count_infosets())
            play_suit = Play(game, mccfr_abs_suit.infoset_data, mccfr.infoset_data)
            result_suit = play_suit.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_suit": infoset_size_suit, "points_suit": result_suit[1], "result_suit": result_suit[0]})

        if abstractions[5]:
            mccfr_abs_suitbet = MCCFR(game, suitbet)
            mccfr_abs_suitbet.train_external(starting_iterations)
            infoset_size_suitbet = sum(mccfr_abs_suitbet.count_infosets())
            play_suitbet = Play(game, mccfr_abs_suitbet.infoset_data, mccfr.infoset_data)
            result_suitbet = play_suitbet.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_suitbet": infoset_size_suitbet, "points_suitbet": result_suitbet[1], "result_suitbet": result_suitbet[0]})

        if abstractions[6]:
            mccfr_abs_adv = MCCFR(game, advanced)
            mccfr_abs_adv.train_external(starting_iterations)
            infoset_size_adv = sum(mccfr_abs_adv.count_infosets())
            play_adv = Play(game, mccfr_abs_adv.infoset_data, mccfr.infoset_data)
            result_adv = play_adv.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_adv": infoset_size_adv, "points_adv": result_adv[1], "result_adv": result_adv[0]})


    for i in tqdm(range(intervals), leave=False):
        mccfr.train_external(iterations_per_interval)
        infoset_size_normal = sum(mccfr.count_infosets())
        wandb.log({"infoset_size_normal": infoset_size_normal, 'iteration': starting_iterations + (i+1)*iterations_per_interval})

        if abstractions[0]:
            mccfr_abs_naive.train_external(iterations_per_interval)
            infoset_size_naive = sum(mccfr_abs_naive.count_infosets())
            play_naive = Play(game, mccfr_abs_naive.infoset_data, mccfr.infoset_data)
            result_naive = play_naive.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_naive": infoset_size_naive, "points_naive": result_naive[1], "result_naive": result_adv[0]})
        
        if abstractions[1]:
            mccfr_abs_sim.train_external(iterations_per_interval)
            infoset_size_sim = sum(mccfr_abs_sim.count_infosets())
            play_sim = Play(game, mccfr_abs_sim.infoset_data, mccfr.infoset_data)
            result_sim = play_sim.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_simple": infoset_size_sim, "points_simple": result_sim[1], "result_simple": result_sim[0]})
        
        if abstractions[2]:
            mccfr_abs_simh.train_external(iterations_per_interval)
            infoset_size_simh = sum(mccfr_abs_simh.count_infosets())
            play_simh = Play(game, mccfr_abs_simh.infoset_data, mccfr.infoset_data)
            result_simh = play_simh.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_simplehand": infoset_size_simh, "points_simplehand": result_simh[1], "result_simplehand": result_simh[0]})
        
        if abstractions[3]:
            mccfr_abs_bets.train_external(iterations_per_interval)
            infoset_size_bets = sum(mccfr_abs_bets.count_infosets())
            play_bets = Play(game, mccfr_abs_bets.infoset_data, mccfr.infoset_data)
            result_bets = play_bets.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_bets": infoset_size_bets, "points_bets": result_bets[1], "result_bets": result_bets[0]})
        
        if abstractions[4]:
            mccfr_abs_suit.train_external(iterations_per_interval)
            infoset_size_suit = sum(mccfr_abs_suit.count_infosets())
            play_suit = Play(game, mccfr_abs_suit.infoset_data, mccfr.infoset_data)
            result_suit = play_suit.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_suit": infoset_size_suit, "points_suit": result_suit[1], "result_suit": result_suit[0]})
        
        if abstractions[5]:
            mccfr_abs_suitbet.train_external(iterations_per_interval)
            infoset_size_suitbet = sum(mccfr_abs_suitbet.count_infosets())
            play_suitbet = Play(game, mccfr_abs_suitbet.infoset_data, mccfr.infoset_data)
            result_suitbet = play_suitbet.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_suitbet": infoset_size_suitbet, "points_suitbet": result_suitbet[1], "result_suitbet": result_suitbet[0]})
        
        if abstractions[6]:
            mccfr_abs_adv.train_external(iterations_per_interval)
            infoset_size_adv = sum(mccfr_abs_adv.count_infosets())
            play_adv = Play(game, mccfr_abs_adv.infoset_data, mccfr.infoset_data)
            result_adv = play_adv.play_n_rounds(eval_iterations)
            wandb.log({"infoset_size_advanced": infoset_size_adv, "points_advanced": result_adv[1], "result_advanced": result_adv[0]})


    if name != '':
        mccfr.save_dict(name + '_normal')
        if abstractions[0]:
            mccfr_abs_naive.save_dict(name + '_naive')
        if abstractions[1]:
            mccfr_abs_sim.save_dict(name + '_simple')
        if abstractions[2]:
            mccfr_abs_simh.save_dict(name + '_simplehand')
        if abstractions[3]:
            mccfr_abs_bets.save_dict(name + '_bets')
        if abstractions[4]:
            mccfr_abs_suit.save_dict(name + '_suit')
        if abstractions[5]:
            mccfr_abs_suitbet.save_dict(name + '_suitbet')
        if abstractions[6]:
            mccfr_abs_adv.save_dict(name + '_advanced')


def abstraction_func(suits, ranks, hand_size, starting_iterations, train_iterations, intervals, eval_iterations, name, abstraction):
    """Function to run the abstraction experiment."""
    iterations_per_interval = int(train_iterations / intervals)
    deck = Deck(suits, ranks)
    game = Game(deck, hand_size)
    mccfr = MCCFR(game, identity)
    mccfr_abs = MCCFR(game, abstraction)
    mccfr1 = MCCFR(game, identity)
    infoset_size_max = len(mccfr1.make_dict())
    
    if starting_iterations == 0:
        infoset_size_normal = sum(mccfr.count_infosets())
        infoset_size_abs = sum(mccfr_abs.count_infosets())
        play_abs = Play(game, mccfr_abs.infoset_data, mccfr.infoset_data)
        result_abs = play_abs.play_n_rounds(eval_iterations)
        wandb.log({f"infoset_size_{abstraction.__name__}": infoset_size_abs,
                   f'infoset_size_normal': infoset_size_normal,
                   f'points_{abstraction.__name__}': result_abs[1],
                   f'result_{abstraction.__name__}': result_abs[0],
                   f'infoset_size_max': {infoset_size_max},
                   'iteration': 0})
    else:
        mccfr.train_external(starting_iterations)
        mccfr_abs.train_external(starting_iterations)
        infoset_size_normal = sum(mccfr.count_infosets())
        infoset_size_abs = sum(mccfr_abs.count_infosets())
        play_abs = Play(game, mccfr_abs.infoset_data, mccfr.infoset_data)
        result_abs = play_abs.play_n_rounds(eval_iterations)
        wandb.log({f"infoset_size_{abstraction.__name__}": infoset_size_abs,
                   f'infoset_size_normal': infoset_size_normal,
                   f'points_{abstraction.__name__}': result_abs[1],
                   f'result_{abstraction.__name__}': result_abs[0],
                   f'infoset_size_max': {infoset_size_max},
                   'iteration': starting_iterations})

    for i in tqdm(range(intervals), leave=False):
        mccfr.train_external(iterations_per_interval)
        mccfr_abs.train_external(iterations_per_interval)
        infoset_size_normal = sum(mccfr.count_infosets())
        infoset_size_abs = sum(mccfr_abs.count_infosets())
        play_abs = Play(game, mccfr_abs.infoset_data, mccfr.infoset_data)
        result_abs = play_abs.play_n_rounds(eval_iterations)
        wandb.log({f"infoset_size_{abstraction.__name__}": infoset_size_abs,
                   f'infoset_size_normal': infoset_size_normal,
                   f'points_{abstraction.__name__}': result_abs[1],
                   f'result_{abstraction.__name__}': result_abs[0],
                   f'infoset_size_max': {infoset_size_max},
                   'iteration': starting_iterations + ((i+1)*iterations_per_interval)})
    if name != '':
        mccfr.save_dict(name + '_normal')
        mccfr_abs.save_dict(name + f"_{abstraction.__name__}")