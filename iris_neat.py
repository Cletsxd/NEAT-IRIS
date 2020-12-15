from __future__ import print_function
import os
import neat
# import visualize
import pandas as pd
import numpy as np
import math

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
Y_real = data.target

Y = [] # 0 = [1, 0, 0], 1 = [0, 1, 0], 2 = [0, 0, 1]

for y_i in Y_real:
    if y_i == 0:
        Y.append([1, 0, 0])
    elif y_i == 1:
        Y.append([0, 1, 0])
    elif y_i == 2:
        Y.append([0, 0, 1])

Y = np.asarray(Y)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        y_pred = []
        for xi, xo in zip(X, Y):
            output = net.activate(xi)

            y_pred.append(np.argmax(np.asarray(output)))

        genome.fitness = accuracy_score(Y_real, y_pred)

        """error_for_output = []

            error_for_output.append((xo - output) ** 2)

        error_for_output = np.asarray(error_for_output)

        genome.fitness -= np.mean(error_for_output)"""

def run(config_file, data_test, file_name_stats):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    ##                           v    number of generations    
    winner = p.run(eval_genomes, 50)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    """for xi, xo in zip(data_inputs, data_outputs):
        output = winner_net.activate(xi)
        print("\ninput {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
        """
    node_names = {
            -1: 'A',
            -2: 'B',
            -3: 'C',
            -4: 'D',
            0:'0', 1: '1', 2: '2'}
    
    #visualize.draw_net(config, winner, True, filename = file_name_stats, node_names=node_names)
    #visualize.plot_stats(stats, ylog=False, view=True, filename = "avg_fitness" + file_name_stats + ".svg")
    #visualize.plot_species(stats, view=True, filename = "speciation" + file_name_stats + ".svg")

    # Running again 10 first generations
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4') # lleva a biblioteca random
    #p.run(eval_genomes, 10)
    
    #print(winner_net.node_evals)
    
    file_node = open("weights_" + file_name_stats, "w")
    file_node.write(str(winner_net.node_evals))
    file_node.close()

    y_pred = []
    for xi in X:
        out = winner_net.activate(xi)
        max_out = max(out)
        out_index = out.index(max_out)
        
        y_pred.append(out_index)
    
    return y_pred

print(run("conf", "", "3"))