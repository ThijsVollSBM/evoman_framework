###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os
from numpy import linalg as LA
import pandas as pd

def cons_multi_patch(values):

        return values


def simulation(env,x,mode):

    f,p,e,t = env.play(pcont=x)

    #f = fitness
    #p = player life
    #e = enemy life
    #t = time

    if mode == 'evaluate':
        return f
    
    if mode == 'gain':
        return f,p,e,t
    

def main(gene):

    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'round_robin_selection'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)


    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.

    gains = []

    
    
    env = Environment(experiment_name=experiment_name,
                    enemies=[1,2,3,4,5,6,7,8],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    multiplemode='yes',
                    level=2,
                    speed="fastest",
                    visuals=False)
    
    env.cons_multi = cons_multi_patch

    for i in range(5):

        _, p, e, _ = simulation(env, gene, mode='gain')

        gain = p-e

        gains.append(gain)


    return np.mean(gains, axis=0)
    

lowerbound = -1
upperbound = 1
population_size = 20



if __name__ == '__main__':

    data = []

    runs = list(range(0,10))
    evals = ['dynamic','static']
    enemy_sets = [1,2]

    for set in enemy_sets:

        for run in runs:

            for eval in evals:

                gene = np.loadtxt(f'best_genes/best_{eval}_eval_gene_enemy_index_{set}_run_nr{run}.txt') 
                mean_gains_arr = main(gene) 
                mean_gains_avg = np.mean(mean_gains_arr)      
                results = list(mean_gains_arr)
                results.append(mean_gains_avg)
                results.append(eval)
                results.append(run)
                results.append(set)
                data.append(results)   


    gene = np.loadtxt(f'genes/77.txt') 
    mean_gains_arr = main(gene) 
    mean_gains_avg = np.mean(mean_gains_arr)      
    results = list(mean_gains_arr)
    results.append(mean_gains_avg)
    results.append('from')
    results.append('other')
    results.append('laptop')
    data.append(results)    


    dataframe = pd.DataFrame(data=data, columns=['1','2','3','4','5','6','7','8','mean','eval','run_nr','enemy_set'])

    dataframe.to_csv('all_gains.csv')