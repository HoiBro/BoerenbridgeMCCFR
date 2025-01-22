from Abstraction_functions import identity, simple, simple_hand, naive, bets, suit, suitbet, advanced
import wandb
from Experiment_functions import full_abstraction, abstraction_func, exploit, fast, retrain
import argparse

"""This is the program used for the MCCFR experiments."""

# Default parameters
suits = 4
ranks = 13
hand_size = 2
starting_iterations = 0
train_iterations = 100000
intervals = 400
eval_iterations = 2500
run_name = 'SuitbetTest'
abstraction = "suitbet"
FLAGS = None

# When the algorithm has no abstraction, choose abstraction = "identity"
train = False
speed = True

# These are the abstractions for the "full" abstraction.
# The abstractions to choose from are naive, simple, simple_hand, bets, suit, suitbet, advanced in that order.
abstractions = [False, False, False, False, False, False, False]

def main():
    abstraction_functions = {
        "identity": identity,
        "adv": advanced,
        "bets": bets,
        "naive": naive,
        "sim": simple,
        "sim_hand": simple_hand,
        "suit": suit,
        "suitbet": suitbet
    }
    if train:
        retrain(FLAGS.suits, FLAGS.ranks, FLAGS.hand_size, FLAGS.train_iterations,
                FLAGS.run_name, abstraction_functions[FLAGS.abstraction])
    elif speed:
        fast(FLAGS.suits, FLAGS.ranks, FLAGS.hand_size, FLAGS.train_iterations,
             FLAGS.run_name, abstraction_functions[FLAGS.abstraction])
    elif abstraction == "":
        exploit(FLAGS.suits, FLAGS.ranks, FLAGS.hand_size, FLAGS.starting_iterations,
                FLAGS.train_iterations, FLAGS.intervals, FLAGS.eval_iterations, FLAGS.run_name)
    elif abstraction == "full":
        full_abstraction(FLAGS.suits, FLAGS.ranks, FLAGS.hand_size, FLAGS.starting_iterations,
                         FLAGS.train_iterations, FLAGS.intervals, FLAGS.eval_iterations, FLAGS.run_name, abstractions)
    else:
        abstraction_func(FLAGS.suits, FLAGS.ranks, FLAGS.hand_size, FLAGS.starting_iterations, FLAGS.train_iterations,
                         FLAGS.intervals, FLAGS.eval_iterations, FLAGS.run_name, abstraction_functions[FLAGS.abstraction])


if __name__ == '__main__':
    if not speed:
        wandb.init(project='BoerenbridgeMCCFR', group='Tests', name=run_name)
        config = wandb.config

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--suits', type=int, default=suits,
                        help='Number of suits')
    parser.add_argument('--ranks', type=int, default=ranks,
                        help='Number of ranks')
    parser.add_argument('--hand_size', type=int, default=hand_size,
                        help='Number of cards in a hand')
    parser.add_argument('--starting_iterations', type=int, default=starting_iterations,
                        help='The amount of iterations to perform before the evaluation process')
    parser.add_argument('--train_iterations', type=int, default=train_iterations,
                        help='Number of total train iterations')
    parser.add_argument('--intervals', type=int, default=intervals,
                        help='Frequency of evaluation')
    parser.add_argument('--eval_iterations', type=int, default=eval_iterations,
                        help='Number of iterations for evaluation')
    parser.add_argument('--run_name', type=str, default=run_name,
                        help='Name for the run/saved infodictionary')
    parser.add_argument('--abstraction', type=str, default=abstraction,
                        help='Abstraction type')
    FLAGS, unparsed = parser.parse_known_args()
    if not speed:
        config.update(FLAGS)

    main()
