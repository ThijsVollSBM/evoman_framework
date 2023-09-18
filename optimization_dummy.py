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

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

# update mutation rate
def update_mutation_rate(sigma, overall_rl, coordinate_rl):

    return sigma * np.exp(overall_rl*np.random.normal()+coordinate_rl*np.random.normal())


def initialize_population(env, experiment_name, lowerbound, upperbound,
                            population_size, n_vars):

    if not os.path.exists(experiment_name+'/evoman_solstate'):

        print( '\nNEW EVOLUTION\n')

        population = np.random.uniform(lowerbound, upperbound, (population_size, n_vars))

        mutation_step_size = np.random.uniform(lowerbound, upperbound, (population_size, 1))

        population_fitness = evaluate(env,population)

        ini_g = 0
        solutions = [population, population_fitness, mutation_step_size]
        env.update_solutions(solutions)

    else:

        print( '\nCONTINUING EVOLUTION\n')

        env.load_state()

        population = env.solutions[0]

        population_fitness = env.solutions[1]

        mutation_step_size = env.solutions[2]

        # finds last generation number
        file_aux  = open(experiment_name+'/gen.txt','r')
        ini_g = int(file_aux.readline())
        file_aux.close()

    return population, population_fitness, mutation_step_size, ini_g


def save_results(experiment_name, ini_g, best, mean, std):
    # saves results for first pop

    print( '\n GENERATION '+str(ini_g)+' '+str(round(best,6))+' '+str(round(mean,6))+' '+str(round(std,6)))

    with open(experiment_name+'/results.txt','a') as file:
        file.write('\n\ngen best mean std')
        file.write('\n'+str(ini_g)+' '+str(round(best,6))+' '+str(round(mean,6))+' '+str(round(std,6))   )


def generate_offspring(population ,mutation_step_sizes, learning_rate_overall, learning_rate_coordinate):

    mutation_step_sizes = update_mutation_rate(sigma=mutation_step_sizes, 
                                                   overall_rl=learning_rate_overall, 
                                                   coordinate_rl=learning_rate_coordinate)

    mutations = mutation_step_sizes*np.random.normal(0,1,(100,1))

    return np.add(population,mutations), mutation_step_sizes



def main():

    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'optimization_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)


    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)


    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    
    # start writing your own code from here

    generations_max = 30
    last_best = 0
    lowerbound = -1
    upperbound = 1
    population_size = 100
    learning_rate_overall = 1/((2*population_size)**0.5)
    learning_rate_coordinate = 1/((2*(population_size**0.5))**0.5)

    population, population_fitness, mutation_step_sizes, generation = initialize_population(env=env, 
                                                                       experiment_name=experiment_name,
                                                                       lowerbound=lowerbound,
                                                                       upperbound=upperbound,
                                                                       population_size=population_size,
                                                                       n_vars=n_vars
                                                                       )

    

    for i in range(generation+1, generations_max):

        best_solution_index = np.argmax(population_fitness)
        mean = np.mean(population_fitness)
        std = np.std(population_fitness)

        best_chromosome = population[best_solution_index]
        best_fitness = population_fitness[best_solution_index]
        best_mutation_rate = mutation_step_sizes[best_solution_index]

        save_results(experiment_name=experiment_name, ini_g=generation, best=best_fitness, mean=mean, std=std)

        #generate offspring (with new mutation step_sizes)
        offspring, offspring_mutation_step_sizes = generate_offspring(population, learning_rate_overall=learning_rate_overall, 
                           learning_rate_coordinate=learning_rate_coordinate, mutation_step_sizes=mutation_step_sizes)
    
        #evaluate new solutions
        offspring_fitness = evaluate(env, offspring)

        #cumulative population
        cum_population = np.vstack((population, offspring))

        #cumulative fitness
        cum_fitness = np.concatenate((population_fitness, offspring_fitness))

        #cumulative mutation step_sizes
        cum_step_sizes = np.concatenate((mutation_step_sizes,offspring_mutation_step_sizes))


        #sort cumulative population and step sizes according to their fitness
        cum_population_sorted = cum_population[cum_fitness.argsort()]
        cum_step_sizes_sorted = cum_step_sizes[cum_fitness.argsort()]
        cum_fitess_sorted = cum_fitness[cum_fitness.argsort()]

        #select top 100 chromosomes and their step sizes and population fitness
        population = cum_population_sorted[population_size:]
        mutation_step_sizes = cum_step_sizes_sorted[population_size:]
        population_fitness = cum_fitess_sorted[population_size:]
        


if __name__ == '__main__':
    main()