import wandb
from Experiment_functions import full_abstraction, abstraction_func, exploit
import argparse

"""This is the program, used for the MCCFR experiments."""

# Default parameters
suits = 2
ranks = 3
hand_size = 2
starting_iterations = 0
train_iterations = 10000
intervals = 400
eval_iterations = 2500
run_name = 'ShortTest'
FLAGS = None


def main():
    exploit(FLAGS.suits, FLAGS.ranks, FLAGS.hand_size, FLAGS.starting_iterations,
            FLAGS.train_iterations, FLAGS.intervals, FLAGS.eval_iterations, FLAGS.run_name)


if __name__ == '__main__':
    wandb.init(project='BoerenbridgeAI', group='Tests', name=run_name)
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
                        help='Name for the run/saved infodict')
    FLAGS, unparsed = parser.parse_known_args()
    config.update(FLAGS)

    main()
