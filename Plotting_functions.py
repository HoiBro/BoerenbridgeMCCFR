import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from Deck import Deck
from Game import Game
from MCCFR import MCCFR
from Abstraction_functions import identity, simple, simple_hand, naive, bets, suit, suitbet, advanced
from Play import Play
import os
import pickle

# Just as a warning, this code is an absolute mess and I could've probably done this WAY better using string manipulation and a for loop but ah well, it is what it is now (I'm sorry :/).

def exploit_plotter(suits, ranks, hand_size, train_iterations, intervals, eval_iterations, name, runs):
    """Function to plot the MCCFR experiment."""
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
    if name != '':
        filename = f"Dicts/{name}_results.pkl"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        a_file = open(filename, "wb")
        pickle.dump(results, a_file)
        pickle.dump(mean, a_file)
        pickle.dump(std, a_file)
        a_file.close()
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


def abstraction_plotter(suits, ranks, hand_size, train_iterations, intervals, eval_iterations, name, abstraction, runs, s):
    """Function to plot the abstraction experiment."""
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
    n_plot = np.linspace(0, train_iterations, intervals+1)
    if s:
        score_mean = np.mean(score, axis=0)
        score_std = np.std(score, axis=0)
        plt.fill_between(n_plot, score_mean + score_std, score_mean - score_std, alpha=0.1, color='g', label='s_std')
        plt.plot(n_plot, score_mean, label='s_gem', color='g')

    results_mean = np.mean(results, axis=0)
    results_std = np.std(results, axis=0)
    plt.fill_between(n_plot, results_mean + results_std, results_mean - results_std, alpha=0.1, color='r', label='r_std')
    plt.plot(n_plot, results_mean, label='r_gem', color='r')
    
    plt.legend()
    plt.title(f"Gemiddelde score tegen niet abstracte strategie")
    plt.xlabel('Iteraties')
    plt.ylabel('Gemiddelde score')
    os.makedirs(os.path.dirname(f"Plots/abstraction/Boerenbridge_abstraction_{suits}_{ranks}_{hand_size}_{train_iterations}_{eval_iterations}_ " +
                                f"{abstraction=}".split('=')[0] + f"{runs}"), exist_ok=True)
    plt.savefig(f"Plots/abstraction/Boerenbridge_abstraction_{suits}_{ranks}_{hand_size}_{train_iterations}_{eval_iterations}_ " +
                f"{abstraction=}".split('=')[0] + f"{runs}")
    plt.show()


def full_abstraction_plotter(suits, ranks, hand_size, train_iterations, intervals, eval_iterations, name, abstractions, runs, s):
    """Function to plot the abstraction experiment using multiple abstraction methods."""
    infoset_sizes_normal =[]
    iterations_per_interval = int(train_iterations / intervals)
    
    if abstractions[0]:
        score_naive = []
        results_naive = []
        infoset_sizes_naive = []
    
    if abstractions[1]:
        score_sim = []
        results_sim = []
        infoset_sizes_sim = []

    if abstractions[2]:
        score_simh = []
        results_simh = []
        infoset_sizes_simh = []

    if abstractions[3]:
        score_bets = []
        results_bets = []
        infoset_sizes_bets = []
    
    if abstractions[4]:
        score_suit = []
        results_suit = []
        infoset_sizes_suit = []

    if abstractions[5]:
        score_suitbet = []
        results_suitbet = []
        infoset_sizes_suitbet = []
    
    if abstractions[6]:
        score_adv = []
        results_adv = []
        infoset_sizes_adv = []


    for i in range(runs):
        deck = Deck(suits, ranks)
        game = Game(deck, hand_size)
        mccfr = MCCFR(game, identity)
        infoset_size_normal = []
        infoset_size_normal.append(sum(mccfr.count_infosets()))

        if abstractions[0]:
            mccfr_abs_naive = MCCFR(game, naive)
            points_naive = []
            result_naive = []
            infoset_size_naive = []
            infoset_size_naive.append(sum(mccfr_abs_naive.count_infosets()))
            play_naive = Play(game, mccfr_abs_naive.infoset_data, mccfr.infoset_data)
            rounds_naive = play_naive.play_n_rounds(eval_iterations)
            points_naive.append(rounds_naive[1])
            result_naive.append(rounds_naive[0])

        if abstractions[1]:
            mccfr_abs_sim = MCCFR(game, simple)
            points_sim = []
            result_sim = []
            infoset_size_sim = []
            infoset_size_sim.append(sum(mccfr_abs_sim.count_infosets()))
            play_sim = Play(game, mccfr_abs_sim.infoset_data, mccfr.infoset_data)
            rounds_sim = play_sim.play_n_rounds(eval_iterations)
            points_sim.append(rounds_sim[1])
            result_sim.append(rounds_sim[0])
        
        if abstractions[2]:
            mccfr_abs_simh = MCCFR(game, simple_hand)
            points_simh = []
            result_simh = []
            infoset_size_simh = []
            infoset_size_simh.append(sum(mccfr_abs_simh.count_infosets()))
            play_simh = Play(game, mccfr_abs_simh.infoset_data, mccfr.infoset_data)
            rounds_simh = play_simh.play_n_rounds(eval_iterations)
            points_simh.append(rounds_simh[1])
            result_simh.append(rounds_simh[0])
        
        if abstractions[3]:
            mccfr_abs_bets = MCCFR(game, bets)
            points_bets = []
            result_bets = []
            infoset_size_bets = []
            infoset_size_bets.append(sum(mccfr_abs_bets.count_infosets()))
            play_bets = Play(game, mccfr_abs_bets.infoset_data, mccfr.infoset_data)
            rounds_bets = play_bets.play_n_rounds(eval_iterations)
            points_bets.append(rounds_bets[1])
            result_bets.append(rounds_bets[0])

        if abstractions[4]:
            mccfr_abs_suit = MCCFR(game, suit)
            points_suit = []
            result_suit = []
            infoset_size_suit = []
            infoset_size_suit.append(sum(mccfr_abs_suit.count_infosets()))
            play_suit = Play(game, mccfr_abs_suit.infoset_data, mccfr.infoset_data)
            rounds_suit = play_suit.play_n_rounds(eval_iterations)
            points_suit.append(rounds_suit[1])
            result_suit.append(rounds_suit[0])
        
        if abstractions[5]:
            mccfr_abs_suitbet = MCCFR(game, suitbet)
            points_suitbet = []
            result_suitbet = []
            infoset_size_suitbet = []
            infoset_size_suitbet.append(sum(mccfr_abs_suitbet.count_infosets()))
            play_suitbet = Play(game, mccfr_abs_suitbet.infoset_data, mccfr.infoset_data)
            rounds_suitbet = play_suitbet.play_n_rounds(eval_iterations)
            points_suitbet.append(rounds_suitbet[1])
            result_suitbet.append(rounds_suitbet[0])

        if abstractions[6]:
            mccfr_abs_adv = MCCFR(game, advanced)
            points_adv = []
            result_adv = []
            infoset_size_adv = []
            infoset_size_adv.append(sum(mccfr_abs_adv.count_infosets()))
            play_adv = Play(game, mccfr_abs_adv.infoset_data, mccfr.infoset_data)
            rounds_adv = play_adv.play_n_rounds(eval_iterations)
            points_adv.append(rounds_adv[1])
            result_adv.append(rounds_adv[0])


        for _ in tqdm(range(intervals), leave=False):
            mccfr.train_external(iterations_per_interval)
            infoset_size_normal.append(sum(mccfr.count_infosets()))

            if abstractions[0]:
                mccfr_abs_naive.train_external(iterations_per_interval)
                infoset_size_naive.append(sum(mccfr_abs_naive.count_infosets()))
                play_naive = Play(game, mccfr_abs_naive.infoset_data, mccfr.infoset_data)
                rounds_naive = play_naive.play_n_rounds(eval_iterations)
                points_naive.append(rounds_naive[1])
                result_naive.append(rounds_naive[0])
            
            if abstractions[1]:
                mccfr_abs_sim.train_external(iterations_per_interval)
                infoset_size_sim.append(sum(mccfr_abs_sim.count_infosets()))
                play_sim = Play(game, mccfr_abs_sim.infoset_data, mccfr.infoset_data)
                rounds_sim = play_sim.play_n_rounds(eval_iterations)
                points_sim.append(rounds_sim[1])
                result_sim.append(rounds_sim[0])
            
            if abstractions[2]:
                mccfr_abs_simh.train_external(iterations_per_interval)
                infoset_size_simh.append(sum(mccfr_abs_simh.count_infosets()))
                play_simh = Play(game, mccfr_abs_simh.infoset_data, mccfr.infoset_data)
                rounds_simh = play_simh.play_n_rounds(eval_iterations)
                points_simh.append(rounds_simh[1])
                result_simh.append(rounds_simh[0])
            
            if abstractions[3]:
                mccfr_abs_bets.train_external(iterations_per_interval)
                infoset_size_bets.append(sum(mccfr_abs_bets.count_infosets()))
                play_bets = Play(game, mccfr_abs_bets.infoset_data, mccfr.infoset_data)
                rounds_bets = play_bets.play_n_rounds(eval_iterations)
                points_bets.append(rounds_bets[1])
                result_bets.append(rounds_bets[0])
            
            if abstractions[4]:
                mccfr_abs_suit.train_external(iterations_per_interval)
                infoset_size_suit.append(sum(mccfr_abs_suit.count_infosets()))
                play_suit = Play(game, mccfr_abs_suit.infoset_data, mccfr.infoset_data)
                rounds_suit = play_suit.play_n_rounds(eval_iterations)
                points_suit.append(rounds_suit[1])
                result_suit.append(rounds_suit[0])
            
            if abstractions[5]:
                mccfr_abs_suitbet.train_external(iterations_per_interval)
                infoset_size_suitbet.append(sum(mccfr_abs_suitbet.count_infosets()))
                play_suitbet = Play(game, mccfr_abs_suitbet.infoset_data, mccfr.infoset_data)
                rounds_suitbet = play_suitbet.play_n_rounds(eval_iterations)
                points_suitbet.append(rounds_suitbet[1])
                result_suitbet.append(rounds_suitbet[0])
            
            if abstractions[6]:
                mccfr_abs_adv.train_external(iterations_per_interval)
                infoset_size_adv.append(sum(mccfr_abs_adv.count_infosets()))
                play_adv = Play(game, mccfr_abs_adv.infoset_data, mccfr.infoset_data)
                rounds_adv = play_adv.play_n_rounds(eval_iterations)
                points_adv.append(rounds_adv[1])
                result_adv.append(rounds_adv[0])


        infoset_sizes_normal.append(infoset_size_normal)
        if name != '':
            mccfr.save_dict(name + '_normal' + str(i))
        
        if abstractions[0]:
            score_naive.append(points_naive)
            results_naive.append(result_naive)
            infoset_sizes_naive.append(infoset_size_naive)
            if name != '':
                mccfr_abs_naive.save_dict(name + '_naive' + str(i))
        
        if abstractions[1]:
            score_sim.append(points_sim)
            results_sim.append(result_sim)
            infoset_sizes_sim.append(infoset_size_sim)
            if name != '':
                mccfr_abs_sim.save_dict(name + '_simple' + str(i))
        
        if abstractions[2]:
            score_simh.append(points_simh)
            results_simh.append(result_simh)
            infoset_sizes_simh.append(infoset_size_simh)
            if name != '':
                mccfr_abs_simh.save_dict(name + '_simplehand' + str(i))

        if abstractions[3]:
            score_bets.append(points_bets)
            results_bets.append(result_bets)
            infoset_sizes_bets.append(infoset_size_bets)
            if name != '':
                mccfr_abs_bets.save_dict(name + '_bets' + str(i))
        
        if abstractions[4]:
            score_suit.append(points_suit)
            results_suit.append(result_suit)
            infoset_sizes_suit.append(infoset_size_suit)
            if name != '':
                mccfr_abs_suit.save_dict(name + '_suit' + str(i))
        
        if abstractions[5]:
            score_suitbet.append(points_suitbet)
            results_suitbet.append(result_suitbet)
            infoset_sizes_suitbet.append(infoset_size_suitbet)
            if name != '':
                mccfr_abs_suitbet.save_dict(name + '_suitbet' + str(i))
        
        if abstractions[6]:
            score_adv.append(points_adv)
            results_adv.append(result_adv)
            infoset_sizes_adv.append(infoset_size_adv)
            if name != '':
                mccfr_abs_adv.save_dict(name + '_advanced' + str(i))


    mccfr.make_dict()
    infoset_max = sum(mccfr.count_infosets())
    n_plot = np.linspace(0, train_iterations, intervals+1)
    plt.figure(figsize=(8, 6))

    if abstractions[0]:
        results_naive = np.array(results_naive)
        results_mean_naive = np.mean(results_naive, axis=0)
        results_std_naive = np.std(results_naive, axis=0)
        plt.fill_between(n_plot, results_mean_naive + results_std_naive, results_mean_naive - results_std_naive, alpha=0.1, color='r', label='result_naive_std')
        plt.plot(n_plot, results_mean_naive, label='result_naive_mean', color='r')
        if s:
            score_naive = np.array(score_naive)
            score_mean_naive = np.mean(score_naive, axis=0)
            score_std_naive = np.std(score_naive, axis=0)
            plt.fill_between(n_plot, score_mean_naive + score_std_naive, score_mean_naive - score_std_naive, alpha=0.1, color='m', label='score_naive_std')
            plt.plot(n_plot, score_mean_naive, label='score_naive_mean', color='m')
    
    if abstractions[1]:
        results_sim = np.array(results_sim)
        results_mean_sim = np.mean(results_sim, axis=0)
        results_std_sim = np.std(results_sim, axis=0)
        plt.fill_between(n_plot, results_mean_sim + results_std_sim, results_mean_sim - results_std_sim, alpha=0.1, color='purple', label='result_sim_std')
        plt.plot(n_plot, results_mean_sim, label='result_sim_mean', color='purple')
        if s:
            score_sim = np.array(score_sim)
            score_std_sim = np.std(score_sim, axis=0)
            score_mean_sim = np.mean(score_sim, axis=0)
            plt.fill_between(n_plot, score_mean_sim + score_std_sim, score_mean_sim - score_std_sim, alpha=0.1, color='mediumvioletred', label='score_sim_std')
            plt.plot(n_plot, score_mean_sim, label='score_sim_mean', color='mediumvioletred')
    
    if abstractions[2]:
        results_simh = np.array(results_simh)
        results_mean_simh = np.mean(results_simh, axis=0)
        results_std_simh = np.std(results_simh, axis=0)
        plt.fill_between(n_plot, results_mean_simh + results_std_simh, results_mean_simh - results_std_simh, alpha=0.1, color='y', label='result_simhand_std')
        plt.plot(n_plot, results_mean_simh, label='result_simhand_mean', color='y')
        if s:
            score_simh = np.array(score_simh)
            score_std_simh = np.std(score_simh, axis=0)
            score_mean_simh = np.mean(score_simh, axis=0)
            plt.fill_between(n_plot, score_mean_simh + score_std_simh, score_mean_simh - score_std_simh, alpha=0.1, color='orange', label='score_simhand_std')
            plt.plot(n_plot, score_mean_simh, label='score_simhand_mean', color='orange')
    
    if abstractions[3]:
        results_bets = np.array(results_bets)
        results_mean_bets = np.mean(results_bets, axis=0)
        results_std_bets = np.std(results_bets, axis=0)
        plt.fill_between(n_plot, results_mean_bets + results_std_bets, results_mean_bets - results_std_bets, alpha=0.1, color='g', label='result_bets_std')
        plt.plot(n_plot, results_mean_bets, label='result_bets_mean', color='g')
        if s:
            score_bets = np.array(score_bets)
            score_std_bets = np.std(score_bets, axis=0)
            score_mean_bets = np.mean(score_bets, axis=0)
            plt.fill_between(n_plot, score_mean_bets + score_std_bets, score_mean_bets - score_std_bets, alpha=0.1, color='lime', label='score_bets_std')
            plt.plot(n_plot, score_mean_bets, label='score_bets_mean', color='lime')
    
    if abstractions[4]:
        results_suit = np.array(results_suit)
        results_mean_suit = np.mean(results_suit, axis=0)
        results_std_suit = np.std(results_suit, axis=0)
        plt.fill_between(n_plot, results_mean_suit + results_std_suit, results_mean_suit - results_std_suit, alpha=0.1, color='b', label='result_suit_std')
        plt.plot(n_plot, results_mean_suit, label='result_suit_mean', color='b')
        if s:
            score_suit = np.array(score_suit)
            score_mean_suit = np.mean(score_suit, axis=0)
            score_std_suit = np.std(score_suit, axis=0)
            plt.fill_between(n_plot, score_mean_suit + score_std_suit, score_mean_suit - score_std_suit, alpha=0.1, color='turquoise', label='score_suit_std')
            plt.plot(n_plot, score_mean_suit, label='score_suit_mean', color='turquoise')
    
    if abstractions[5]:
        results_suitbet = np.array(results_suitbet)
        results_mean_suitbet = np.mean(results_suitbet, axis=0)
        results_std_suitbet = np.std(results_suitbet, axis=0)
        plt.fill_between(n_plot, results_mean_suitbet + results_std_suitbet, results_mean_suitbet - results_std_suitbet, alpha=0.1, color='teal', label='result_suitbet_std')
        plt.plot(n_plot, results_mean_suitbet, label='result_suitbet_mean', color='teal')
        if s:
            score_suitbet = np.array(score_suitbet)
            score_mean_suitbet = np.mean(score_suitbet, axis=0)
            score_std_suitbet = np.std(score_suitbet, axis=0)
            plt.fill_between(n_plot, score_mean_suitbet + score_std_suitbet, score_mean_suitbet - score_std_suitbet, alpha=0.1, color='mediumseagreen', label='score_suitbet_std')
            plt.plot(n_plot, score_mean_suitbet, label='score_suitbet_mean', color='mediumseagreen')
    
    if abstractions[6]:
        results_adv = np.array(results_adv)
        results_mean_adv = np.mean(results_adv, axis=0)
        results_std_adv = np.std(results_adv, axis=0)
        plt.fill_between(n_plot, results_mean_adv + results_std_adv, results_mean_adv - results_std_adv, alpha=0.1, color='c', label='result_adv_std')
        plt.plot(n_plot, results_mean_adv, label='result_adv_mean', color='c')
        if s:
            score_adv = np.array(score_adv)
            score_mean_adv = np.mean(score_adv, axis=0)
            score_std_adv = np.std(score_adv, axis=0)
            plt.fill_between(n_plot, score_mean_adv + score_std_adv, score_mean_adv - score_std_adv, alpha=0.1, color='lightskyblue', label='score_adv_std')
            plt.plot(n_plot, score_mean_adv, label='score_adv_mean', color='lightskyblue')


    plt.legend()
    plt.title(f"Gemiddelde score tegen de niet abstracte strategie")
    plt.xlabel('Iteraties')
    plt.ylabel('Gemiddelde score')
    os.makedirs(os.path.dirname(f"Plots/abstraction/Boerenbridge_abstraction_{suits}_{ranks}_{hand_size}_{train_iterations}_{eval_iterations}_{runs}"), exist_ok=True)
    plt.savefig(f"Plots/abstraction/Boerenbridge_abstraction_{suits}_{ranks}_{hand_size}_{train_iterations}_{eval_iterations}_{runs}")
    plt.show()

    results_normal = np.array(infoset_sizes_normal)
    n_plot = np.linspace(0, train_iterations, intervals + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(n_plot, np.repeat(infoset_max, intervals + 1), '--', label='Totaal Informatie Sets', color='m')
    mean_normal = np.mean(results_normal, axis=0)
    plt.plot(n_plot, mean_normal, label='Identity', color='gray')

    if abstractions[0]:
        results_naive = np.array(infoset_sizes_naive)
        mean_naive = np.mean(results_naive, axis=0)
        plt.plot(n_plot, mean_naive, label='Naive', color='r')
    
    if abstractions[1]:
        results_sim = np.array(infoset_sizes_sim)
        mean_sim = np.mean(results_sim, axis=0)
        plt.plot(n_plot, mean_sim, label='Simple', color='purple')
    
    if abstractions[2]:
        results_simh = np.array(infoset_sizes_simh)
        mean_simh = np.mean(results_simh, axis=0)
        plt.plot(n_plot, mean_simh, label='Simple Hand', color='y')
    
    if abstractions[3]:
        results_bets = np.array(infoset_sizes_bets)
        mean_bets = np.mean(results_bets, axis=0)
        plt.plot(n_plot, mean_bets, label='Bets', color='g')
    
    if abstractions[4]:
        results_suit = np.array(infoset_sizes_suit)
        mean_suit = np.mean(results_suit, axis=0)
        plt.plot(n_plot, mean_suit, label='Suit', color='b')
    
    if abstractions[5]:
        results_suitbet = np.array(infoset_sizes_suitbet)
        mean_suitbet = np.mean(results_suitbet, axis=0)
        plt.plot(n_plot, mean_suitbet, label='Suitbet', color='teal')
    
    if abstractions[6]:
        results_adv = np.array(infoset_sizes_adv)
        mean_adv = np.mean(results_adv, axis=0)
        plt.plot(n_plot, mean_adv, label='Advanced', color='c')


    plt.legend()
    plt.title(f"Gemiddeld aantal informatie sets behaald")
    plt.xlabel('Iteraties')
    plt.ylabel('Informatie sets')
    os.makedirs(os.path.dirname(f"Plots/abstraction/Boerenbridge_abstraction_infosize_{suits}_{ranks}_{hand_size}_{hand_size}_{train_iterations}_{eval_iterations}_{runs}"), exist_ok=True)
    plt.savefig(f"Plots/abstraction/Boerenbridge_abstraction_infosize_{suits}_{ranks}_{hand_size}_{hand_size}_{train_iterations}_{eval_iterations}_{runs}")
    plt.show()
