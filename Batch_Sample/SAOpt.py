import os
import sys
import time
import logging as log
import numpy as np
import argparse
from collections import namedtuple, Counter


def main():
    parser = argparse.ArgumentParser(prog='SAOpt.py')
    parser.set_defaults(func=main_sa)
    get_argparser(parser)
    # Display help if no arguments are given
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    # Recover arguments
    args = parser.parse_args()
    # Execute the command
    args.func(args, parser)


def main_sa(args, argparser):
    time_start = time.time()
    # Configure logging level based on verbosity
    if args.verbose:
        log.basicConfig(level=log.INFO, format="VERBOSE: %(message)s")
    if args.debug:
        log.basicConfig(level=log.DEBUG, format="DEBUG: %(message)s")
    # Print all arguments
    for k, v in vars(args).items():
        sys.stdout.write("#" + str(k) + ':' + str(v) + "\n")
    # Set the random seed
    np.random.seed(args.rand_seed)
    log.info(f"Initialized random seed: {args.rand_seed}")
    # Global matrix variable for easy access
    global matrix
    meta = parse_input(args.input)
    matrix = meta.matrix
    log.info(f"Loaded matrix with shape: {matrix.shape}")
    # Choose the initial path: greedy or random
    if args.greedy_init:
        log.info("Using greedy strategy for initial solution.")
        initial_path = init_path(rand=False)
    else:
        log.info("Using random initial solution.")
        initial_path = init_path(rand=True)
    log.info(f"Starting simulated annealing...")
    path, trace = simulated_annealing(
        initial_path,
        args.init_temp,
        args.final_temp,
        args.cooling_rate,
        args.iterations
    )
    tgt_mean, tgt_var = target_function(path)
    # Log the results
    log.info("Optimal path: {}".format('->'.join(['[{}:{}]'.format(i, meta.softwares[j]) for i, j in path])))
    log.info(f"Optimal path mean: {tgt_mean}")
    log.info(f"Optimal path variance: {tgt_var}")
    log.info("Software usage count: {}".format(Counter([meta.softwares[i[1]] for i in path])))
    # Optionally plot the trace
    if args.graphical:
        plot_trace(trace)
    # Optionally write the output matrix to a file
    if args.output:
        with open(args.output, 'w') as fo:
            fo.write('ID\t' + '\t'.join(meta.softwares) + '\n')
            for sample, X in zip(meta.samples, output_matrix(path)):
                fo.write(sample + '\t' + '\t'.join(map(str, X)) + '\n')
    log.info(f"Completed.")
    sys.stdout.write("#Elapsed time:" + str(time.time() - time_start) + "\n")


def parse_input(input_file):
    """
    Parse the input file and return the matrix, software names, and sample IDs.
    The input file is expected to be a tab-delimited file with the first row
    containing software names and the first column containing sample IDs.
    """
    # Load the matrix data (skip the header row and the first column)
    matrix = np.genfromtxt(
        fname=input_file,
        delimiter="\t",
        skip_header=1,
        filling_values=np.nan
    )[:, 1:]
    # Read the header to get software names
    with open(input_file) as f:
        header = f.readline().strip('\n').split('\t')[1:]
    softwares = header
    # Read the sample IDs from the first column
    samples = [line.strip('\n').split('\t')[0] for line in open(input_file).readlines()][1:]
    Meta = namedtuple('Meta', ['matrix', 'softwares', 'samples'])
    return Meta(matrix=matrix, softwares=softwares, samples=samples)


def output_matrix(path):
    """
    Generate the output matrix based on the path.
    Each row corresponds to a sample, and a value of 1 indicates
    that a particular software was selected for that sample.
    """
    output = np.zeros_like(matrix)
    for i, j in path:
        output[i][j] = 1
    return output


def get_argparser(parser):
    """
    Define the command-line arguments for the script.
    """
    parser.add_argument(
        "-i", "--input",
        help="Input file (default: -i deviation.tsv)",
        required=True
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file (default: -o sparse.tsv)"
    )
    parser.add_argument(
        "--init-temp",
        help="Initial temperature (default: --init-temp 1.0)",
        action="store",
        nargs='?',
        default=1.0,
        type=float
    )
    parser.add_argument(
        "--final-temp",
        help="Final temperature (default: --final-temp 1e-7)",
        action="store",
        nargs='?',
        default=1e-7,
        type=float
    )
    parser.add_argument(
        "--cooling-rate",
        help="Cooling rate (default: --cooling-rate 0.8)",
        action="store",
        nargs='?',
        default=0.8,
        type=float
    )
    parser.add_argument(
        "--rand-seed",
        help="Random seed (default: --rand-seed 42)",
        action="store",
        nargs='?',
        default=42,
        type=int
    )
    parser.add_argument(
        "--iterations",
        help="Number of iterations per temperature (default: --iterations 100)",
        action="store",
        nargs='?',
        default=100,
        type=int
    )
    parser.add_argument(
        "-g", "--graphical",
        help="Display coverage graph.",
        action="store_true"
    )
    parser.add_argument(
        "--greedy-init",
        help="Use greedy strategy for initial solution.",
        action="store_true"
    )
    parser.add_argument(
        "-v", "--verbose",
        help="Get more information.",
        action="store_true"
    )
    parser.add_argument(
        "-vv", "--debug",
        help="Get detailed debug information.",
        action="store_true"
    )


def init_path(rand=True):
    """
    Initialize the solution path.
    If rand=True, generate a random initial path.
    If rand=False, use the greedy strategy to generate the initial path.
    The greedy strategy selects the software (column) with the minimal value
    for each sample (row).
    """
    if rand:
        # Randomly select a software for each sample
        return [(i, np.random.randint(matrix.shape[1])) for i in range(matrix.shape[0])]
    else:
        # Greedy strategy: select the software with the minimal value for each sample
        min_indices = np.nanargmin(matrix, axis=1)  # Use nanargmin to handle NaNs
        return list(zip(range(len(matrix)), min_indices))


def target_function(path):
    """
    Calculate the mean and variance of the values along the given path.
    The path is a list of (row_index, column_index) tuples.
    """
    values = [matrix[i][j] for i, j in path]
    return np.nanmean(values), np.nanvar(values)  # Use nanmean and nanvar to handle NaNs


def generate_new_path(path):
    """
    Generate a new path by randomly changing the software (column) for a random sample (row).
    This function creates a neighbor solution in the solution space.
    """
    new_path = path.copy()
    # Randomly select a row index (sample)
    row = np.random.randint(matrix.shape[0])
    # Randomly select a new column index (software)
    new_column = np.random.randint(matrix.shape[1])
    new_path[row] = (row, new_column)
    return new_path


def simulated_annealing(path, initial_temp, final_temp, cooling_rate, iterations=100):
    """
    Perform the simulated annealing algorithm to optimize the path.
    Args:
        path: Initial solution path.
        initial_temp: Initial temperature.
        final_temp: Final temperature at which the algorithm stops.
        cooling_rate: Rate at which the temperature decreases.
        iterations: Number of iterations to perform at each temperature level.
    Returns:
        Best path found and the trace of mean and variance values.
    """
    current_temp = initial_temp
    current_path = path
    current_mean, current_var = target_function(current_path)
    best_path = current_path
    best_mean = current_mean
    best_var = current_var
    trace = []
    iteration_counter = 0
    while current_temp > final_temp:
        for _ in range(iterations):
            m, v = target_function(current_path)
            trace.append((m, v, iteration_counter))
            iteration_counter += 1
            # Log the current state
            log.debug(f"Iteration {iteration_counter}, Mean: {m}, Variance: {v}")
            # Generate a new candidate solution
            new_path = generate_new_path(current_path)
            new_mean, new_var = target_function(new_path)
            # Calculate the change in the objective function
            delta = (current_mean - new_mean) + (current_var - new_var)
            # Decide whether to accept the new solution
            if new_mean < current_mean and new_var < current_var:
                # Accept the new solution if it's better
                current_path = new_path
                current_mean, current_var = new_mean, new_var
                # Update the best found solution
                if new_mean < best_mean and new_var < best_var:
                    best_path = new_path
                    best_mean, best_var = new_mean, new_var
            else:
                # Accept the new solution with a probability based on temperature
                accept_prob = np.exp(delta / current_temp)
                if np.random.rand() < accept_prob:
                    current_path = new_path
                    current_mean, current_var = new_mean, new_var
        # Decrease the temperature
        current_temp *= cooling_rate
        log.debug(f"Temperature decreased to {current_temp}")
    log.info(f"Total iterations: {iteration_counter}")
    return best_path, trace


def plot_trace(trace):
    """
    Plot the trace of the mean and variance values over iterations.
    """
    import matplotlib.pyplot as plt
    # Extract data from trace
    means = [item[0] for item in trace]
    vars_ = [item[1] for item in trace]
    iterations = [item[2] for item in trace]
    # Create a 2D plot of mean and variance over iterations
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, means, label='Mean')
    plt.plot(iterations, vars_, label='Variance')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Trace of Mean and Variance over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Optionally save the plot
    # plt.savefig("trace.png")


if __name__ == '__main__':
    main()

