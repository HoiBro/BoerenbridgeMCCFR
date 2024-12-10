from Abstraction_functions import identity, simple, simple_hand, naive, advanced
from Plotting_functions import exploit_plotter, abstraction_plotter, full_abstraction_plotter
import argparse

"""This is the program, used for the Plotting experiments."""

# Default parameters
suits = 2
ranks = 3
hand_size = 2
starting_iterations = 0
train_iterations = 5000
intervals = 1
eval_iterations = 1000
run_name = ''
abstraction = ""
amount = 5
FLAGS = None


def main():
    abstraction_functions = {
        "adv": advanced,
        "sim": simple,
        "sim_hand": simple_hand,
        "naive": naive
    }
    if abstraction == '':
        full_abstraction_plotter(FLAGS.suits, FLAGS.ranks, FLAGS.hand_size, FLAGS.starting_iterations,
                        FLAGS.train_iterations, FLAGS.intervals, FLAGS.eval_iterations, FLAGS.run_name, FLAGS.amount)
    else:
        abstraction_plotter(FLAGS.suits, FLAGS.ranks, FLAGS.hand_size, FLAGS.starting_iterations, FLAGS.train_iterations,
                            FLAGS.intervals, FLAGS.eval_iterations, FLAGS.run_name, abstraction_functions[FLAGS.abstraction], FLAGS.amount)


if __name__ == '__main__':
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
    parser.add_argument('--abstraction', type=str, default=abstraction,
                        help='Abstraction type')
    parser.add_argument('--amount', type=int, default=amount,
                        help='Amount of runs')
    FLAGS, unparsed = parser.parse_known_args()

    main()
