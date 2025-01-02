import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from Deck import Deck
from Game import Game
from MCCFR import MCCFR
from Abstraction_functions import simple, identity, naive, suit, advanced
from Play import Play
import os


def exploit_plotter(suits, ranks, hand_size, starting_iterations, train_iterations, intervals, eval_iterations, name, runs):
    """Function to plot the MCCFR experiment.
    train_iterations: total number of iterations
    interval: number of intevals for eval
    eval_iteration: number of iterations for each eval
    runs: total number of runs"""
    results = []
    iterations_per_interval = int(train_iterations / intervals)
    for i in range(runs):
        result = []
        deck = Deck(suits, ranks)
        game = Game(deck, hand_size)
        mccfr = MCCFR(game, identity)
        result.append(mccfr.get_exploitability(eval_iterations))

        for _ in tqdm(range(intervals), leave=False):
            mccfr.train_external(iterations_per_interval)
            result.append(mccfr.get_exploitability(eval_iterations))
        results.append(result)
        if name != '':
            mccfr.save_dict(name + "_" + str(i))
    results = np.array(results)
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    n_plot = np.linspace(0, train_iterations, intervals+1)
    plt.fill_between(n_plot, mean + std, mean - std, alpha=0.1, color='r', label='Standaard afwijking')
    plt.plot(n_plot, mean, label='Mean', color='r')
    plt.legend()
    plt.title(f"Exploiteerbaarheid for game")
    plt.xlabel('Iteraties')
    plt.ylabel('Exploiteerbaarheid')
    os.makedirs(os.path.dirname(f"Plots/Boerenbridge_exploit_{suits}_{ranks}_{hand_size}_{train_iterations}_{eval_iterations}_{runs}"), exist_ok=True)
    plt.savefig(f"Plots/Boerenbridge_exploit_{suits}_{ranks}_{hand_size}_{train_iterations}_{eval_iterations}_{runs}")
    plt.show()


def abstraction_plotter(suits, ranks, hand_size, starting_iterations, train_iterations, intervals, eval_iterations, name, abstraction, runs):
    """Function to plot the abstraction experiment.
    train_iterations: total number of iterations
    interval: number of intevals for eval
    eval_iteration: number of iterations for each eval
    runs: total number of runs"""
    score = []
    results = []
    iterations_per_interval = int(train_iterations / intervals)
    for i in range(runs):
        points = []
        result = []
        deck = Deck(suits, ranks)
        game = Game(deck, hand_size)
        mccfr = MCCFR(game, identity)
        mccfr_abs = MCCFR(game, abstraction)

        play = Play(game, mccfr_abs.infoset_data, mccfr.infoset_data)
        rounds = play.play_n_rounds(eval_iterations)
        points.append(rounds[1])
        result.append(rounds[0])

        for _ in tqdm(range(intervals), leave=False):
            mccfr.train_external(iterations_per_interval)
            mccfr_abs.train_external(iterations_per_interval)
            play = Play(game, mccfr_abs.infoset_data, mccfr.infoset_data)
            rounds = play.play_n_rounds(eval_iterations)
            points.append(rounds[1])
            result.append(rounds[0])
        score.append(points)
        results.append(result)
        if name != '':
            mccfr.save_dict(name + "_normal" + str(i))
            mccfr_abs.save_dict(name + f"_{abstraction.__name__}" + str(i))
    score = np.array(score)
    results = np.array(results)
    score_mean = np.mean(score, axis=0)
    results_mean = np.mean(results, axis=0)
    score_std = np.std(score, axis=0)
    results_std = np.std(results, axis=0)
    n_plot = np.linspace(0, train_iterations, intervals+1)
    plt.fill_between(n_plot, score_mean + score_std, score_mean - score_std, alpha=0.1, color='g', label='s_std')
    plt.fill_between(n_plot, results_mean + results_std, results_mean - results_std, alpha=0.1, color='r', label='r_std')
    plt.plot(n_plot, score_mean, label='s_mean', color='g')
    plt.plot(n_plot, results_mean, label='r_mean', color='r')
    plt.legend()
    plt.title(f"Gemiddelde score tegen niet abstracte strategie")
    plt.xlabel('Iteraties')
    plt.ylabel('Gemiddelde score')
    os.makedirs(os.path.dirname(f"Plots/abstraction/Boerenbridge_abstraction_{suits}_{ranks}_{hand_size}_{train_iterations}_{eval_iterations}_ " +
                                f"{abstraction=}".split('=')[0] + f"{runs}"), exist_ok=True)
    plt.savefig(f"Plots/abstraction/Boerenbridge_abstraction_{suits}_{ranks}_{hand_size}_{train_iterations}_{eval_iterations}_ " +
                f"{abstraction=}".split('=')[0] + f"{runs}")
    plt.show()


def full_abstraction_plotter(suits, ranks, hand_size, starting_iterations, train_iterations, intervals, eval_iterations, name, runs):
    """Function to plot the abstraction experiment using multiple abstraction methods.
    train_iterations: total number of iterations
    interval: number of intevals for eval
    eval_iteration: number of iterations for each eval
    runs: total number of runs"""
    score_naive = []
    score_sim = []
    score_suit = []
    score_adv = []

    results_naive = []
    results_sim = []
    results_suit = []
    results_adv = []

    infoset_sizes_normal =[]
    infoset_sizes_naive = []
    infoset_sizes_sim = []
    infoset_sizes_suit = []
    infoset_sizes_adv = []

    iterations_per_interval = int(train_iterations / intervals)
    for i in range(runs):
        deck = Deck(suits, ranks)
        game = Game(deck, hand_size)
        mccfr = MCCFR(game, identity)
        mccfr_abs_naive = MCCFR(game, naive)
        mccfr_abs_sim = MCCFR(game, simple)
        mccfr_abs_suit = MCCFR(game, suit)
        mccfr_abs_adv = MCCFR(game, advanced)

        points_naive = []
        points_sim = []
        points_suit = []
        points_adv = []

        result_naive = []
        result_sim = []
        result_suit = []
        result_adv = []

        infoset_size_normal = []
        infoset_size_naive = []
        infoset_size_sim = []
        infoset_size_suit = []
        infoset_size_adv = []

        infoset_size_normal.append(sum(mccfr.count_infosets()))
        infoset_size_naive.append(sum(mccfr_abs_naive.count_infosets()))
        infoset_size_sim.append(sum(mccfr_abs_sim.count_infosets()))
        infoset_size_suit.append(sum(mccfr_abs_suit.count_infosets()))
        infoset_size_adv.append(sum(mccfr_abs_adv.count_infosets()))

        play_naive = Play(game, mccfr_abs_naive.infoset_data, mccfr.infoset_data)
        play_sim = Play(game, mccfr_abs_sim.infoset_data, mccfr.infoset_data)
        play_suit = Play(game, mccfr_abs_suit.infoset_data, mccfr.infoset_data)
        play_adv = Play(game, mccfr_abs_adv.infoset_data, mccfr.infoset_data)

        rounds_naive = play_naive.play_n_rounds(eval_iterations)
        rounds_sim = play_sim.play_n_rounds(eval_iterations)
        rounds_suit = play_suit.play_n_rounds(eval_iterations)
        rounds_adv = play_adv.play_n_rounds(eval_iterations)

        points_naive.append(rounds_naive[1])
        points_sim.append(rounds_sim[1])
        points_suit.append(rounds_suit[1])
        points_adv.append(rounds_adv[1])

        result_naive.append(rounds_naive[0])
        result_sim.append(rounds_sim[0])
        result_suit.append(rounds_suit[0])
        result_adv.append(rounds_adv[0])

        for _ in tqdm(range(intervals), leave=False):
            mccfr.train_external(iterations_per_interval)
            mccfr_abs_naive.train_external(iterations_per_interval)
            mccfr_abs_sim.train_external(iterations_per_interval)
            mccfr_abs_suit.train_external(iterations_per_interval)
            mccfr_abs_adv.train_external(iterations_per_interval)

            infoset_size_normal.append(sum(mccfr.count_infosets()))
            infoset_size_naive.append(sum(mccfr_abs_naive.count_infosets()))
            infoset_size_sim.append(sum(mccfr_abs_sim.count_infosets()))
            infoset_size_suit.append(sum(mccfr_abs_suit.count_infosets()))
            infoset_size_adv.append(sum(mccfr_abs_adv.count_infosets()))

            play_naive = Play(game, mccfr_abs_naive.infoset_data, mccfr.infoset_data)
            play_sim = Play(game, mccfr_abs_sim.infoset_data, mccfr.infoset_data)
            play_suit = Play(game, mccfr_abs_suit.infoset_data, mccfr.infoset_data)
            play_adv = Play(game, mccfr_abs_adv.infoset_data, mccfr.infoset_data)

            rounds_naive = play_naive.play_n_rounds(eval_iterations)
            rounds_sim = play_sim.play_n_rounds(eval_iterations)
            rounds_suit = play_suit.play_n_rounds(eval_iterations)
            rounds_adv = play_adv.play_n_rounds(eval_iterations)

            points_naive.append(rounds_naive[1])
            points_sim.append(rounds_sim[1])
            points_suit.append(rounds_suit[1])
            points_adv.append(rounds_adv[1])

            result_naive.append(rounds_naive[0])
            result_sim.append(rounds_sim[0])
            result_suit.append(rounds_suit[0])
            result_adv.append(rounds_adv[0])
        
        score_naive.append(points_naive)
        score_sim.append(points_sim)
        score_suit.append(points_suit)
        score_adv.append(points_adv)

        results_naive.append(result_naive)
        results_sim.append(result_sim)
        results_suit.append(result_suit)
        results_adv.append(result_adv)

        infoset_sizes_normal.append(infoset_size_normal)
        infoset_sizes_naive.append(infoset_size_naive)
        infoset_sizes_sim.append(infoset_size_sim)
        infoset_sizes_suit.append(infoset_size_suit)
        infoset_sizes_adv.append(infoset_size_adv)

        if name != '':
            mccfr.save_dict(name + '_normal' + str(i))
            mccfr_abs_adv.save_dict(name + '_abs' + str(i))
            mccfr_abs_naive.save_dict(name + '_naive' + str(i))
            mccfr_abs_sim.save_dict(name + '_simple' + str(i))
            mccfr_abs_suit.save_dict(name + '_suit' + str(i))

    mccfr.evaluate()
    infoset_max = sum(mccfr.count_infosets())

    score_naive = np.array(score_naive)
    score_sim = np.array(score_sim)
    score_suit = np.array(score_suit)
    score_adv = np.array(score_adv)

    results_naive = np.array(results_naive)
    results_sim = np.array(results_sim)
    results_suit = np.array(results_suit)
    results_adv = np.array(results_adv)

    n_plot = np.linspace(0, train_iterations, intervals+1)
    plt.figure(figsize=(8, 6))

    score_mean_naive = np.mean(score_naive, axis=0)
    results_mean_naive = np.mean(results_naive, axis=0)
    score_std_naive = np.std(score_naive, axis=0)
    results_std_naive = np.std(results_naive, axis=0)
    plt.fill_between(n_plot, score_mean_naive + score_std_naive, score_mean_naive - score_std_naive, alpha=0.1, color='magenta', label='score_naive_std')
    plt.fill_between(n_plot, results_mean_naive + results_std_naive, results_mean_naive - results_std_naive, alpha=0.1, color='r', label='result_naive_std')
    plt.plot(n_plot, score_mean_naive, label='score_naive_mean', color='magenta')
    plt.plot(n_plot, results_mean_naive, label='result_naive_mean', color='r')

    score_mean_sim = np.mean(score_sim, axis=0)
    results_mean_sim = np.mean(results_sim, axis=0)
    score_std_sim = np.std(score_sim, axis=0)
    results_std_sim = np.std(results_sim, axis=0)
    plt.fill_between(n_plot, score_mean_sim + score_std_sim, score_mean_sim - score_std_sim, alpha=0.1, color='lime', label='score_sim_std')
    plt.fill_between(n_plot, results_mean_sim + results_std_sim, results_mean_sim - results_std_sim, alpha=0.1, color='g', label='result_sim_std')
    plt.plot(n_plot, score_mean_sim, label='score_sim_mean', color='lime')
    plt.plot(n_plot, results_mean_sim, label='result_sim_mean', color='g')

    score_mean_suit = np.mean(score_suit, axis=0)
    results_mean_suit = np.mean(results_suit, axis=0)
    score_std_suit = np.std(score_suit, axis=0)
    results_std_suit = np.std(results_suit, axis=0)
    plt.fill_between(n_plot, score_mean_suit + score_std_suit, score_mean_suit - score_std_suit, alpha=0.1, color='turquoise', label='score_suit_std')
    plt.fill_between(n_plot, results_mean_suit + results_std_suit, results_mean_suit - results_std_suit, alpha=0.1, color='b', label='result_suit_std')
    plt.plot(n_plot, score_mean_suit, label='score_suit_mean', color='turquoise')
    plt.plot(n_plot, results_mean_suit, label='result_suit_mean', color='b')

    score_mean_adv = np.mean(score_adv, axis=0)
    results_mean_adv = np.mean(results_adv, axis=0)
    score_std_adv = np.std(score_adv, axis=0)
    results_std_adv = np.std(results_adv, axis=0)
    plt.fill_between(n_plot, score_mean_adv + score_std_adv, score_mean_adv - score_std_adv, alpha=0.1, color='orange', label='score_adv_std')
    plt.fill_between(n_plot, results_mean_adv + results_std_adv, results_mean_adv - results_std_adv, alpha=0.1, color='y', label='result_adv_std')
    plt.plot(n_plot, score_mean_adv, label='score_adv_mean', color='orange')
    plt.plot(n_plot, results_mean_adv, label='result_adv_mean', color='y')

    plt.legend()
    plt.title(f"Gemiddelde score tegen de niet abstracte strategie")
    plt.xlabel('Iteraties')
    plt.ylabel('Gemiddelde score')
    os.makedirs(os.path.dirname(f"Plots/abstraction/Boerenbridge_abstraction_{suits}_{ranks}_{hand_size}_{train_iterations}_{eval_iterations}_{runs}"), exist_ok=True)
    plt.savefig(f"Plots/abstraction/Boerenbridge_abstraction_{suits}_{ranks}_{hand_size}_{train_iterations}_{eval_iterations}_{runs}")
    plt.show()

    results_normal = np.array(infoset_sizes_normal)
    results_naive = np.array(infoset_sizes_naive)
    results_sim = np.array(infoset_sizes_sim)
    results_suit = np.array(infoset_sizes_suit)
    results_adv = np.array(infoset_sizes_adv)

    n_plot = np.linspace(0, train_iterations, intervals + 1)
    plt.figure(figsize=(8, 6))

    plt.plot(n_plot, np.repeat(infoset_max, intervals + 1), '--', label='Totaal Informatie Sets', color='m')

    mean_normal = np.mean(results_normal, axis=0)
    plt.plot(n_plot, mean_normal, label='Identity', color='c')

    mean_naive = np.mean(results_naive, axis=0)
    plt.plot(n_plot, mean_naive, label='Naive', color='r')

    mean_sim = np.mean(results_sim, axis=0)
    plt.plot(n_plot, mean_sim, label='Simple', color='g')

    mean_suit = np.mean(results_suit, axis=0)
    plt.plot(n_plot, mean_suit, label='Suit', color='b')

    mean_adv = np.mean(results_adv, axis=0)
    plt.plot(n_plot, mean_adv, label='Advanced', color='y')

    plt.legend()
    plt.title(f"Gemiddeld aantal informatie sets behaald")
    plt.xlabel('Iteraties')
    plt.ylabel('Informatie sets')
    os.makedirs(os.path.dirname(f"Plots/abstraction/Boerenbridge_abstraction_infosize_{suits}_{ranks}_{hand_size}_{hand_size}_{train_iterations}_{eval_iterations}_{runs}"), exist_ok=True)
    plt.savefig(f"Plots/abstraction/Boerenbridge_abstraction_infosize_{suits}_{ranks}_{hand_size}_{hand_size}_{train_iterations}_{eval_iterations}_{runs}")
    plt.show()
