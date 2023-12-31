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
def update_mutation_rate(mutation_rates, overall_LR, coordinate_LR):

    step_overall_LR = overall_LR*np.random.normal(0,1)

    step_coordinate_LR = coordinate_LR*np.random.normal(0,1,(100,1))

    return mutation_rates * np.exp(step_overall_LR+step_coordinate_LR)


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

def mutate_population(population ,mutation_step_sizes, learning_rate_overall, learning_rate_coordinate):

    offspring_mutation_rates = update_mutation_rate(mutation_rates=mutation_step_sizes, 
                                                   overall_LR=learning_rate_overall, 
                                                   coordinate_LR=learning_rate_coordinate)

    mutants = np.add(population,offspring_mutation_rates*np.random.normal(0,1,(100,1)))

    for i in range(len(mutants)):

        mutation_prob = np.random.uniform()

        if mutation_prob < MUTATION_PROBABILITY:

            population[i] = mutants[i]
            mutation_step_sizes[i] = offspring_mutation_rates[i]

    return population, mutation_step_sizes

def round_robin(population, mutation_step_sizes, population_fitness):

    scores = np.zeros(population_fitness.shape)

    for i in range(len(population)):
        
        fitness = population_fitness[i]

        enemies = np.random.randint(low=1, high=len(population_fitness),size=10)

        score = (population_fitness[enemies] < fitness).sum()

        scores[i] = score
      
    indices = np.argsort(scores)

    return population[indices[-100:]], mutation_step_sizes[indices[-100:]], population_fitness[indices[-100:]]

def crossover(population, mutation_step_sizes, population_fitness):

    offspring = []

    offspring_mutation_rates = []

    for i in range(0,population_size, 2):

        p1_index = tournament(population,population_fitness)
        p2_index = tournament(population,population_fitness)

        p1 = population[p1_index].copy()
        p1_sigma = mutation_step_sizes[p1_index]

        p2 = population[p2_index].copy()
        p2_sigma = mutation_step_sizes[p2_index]

        bools = np.random.choice(a=[False, True], size=p1.shape)

        for i in range(len(bools)):
            
            #if True, swap the genome of the two parents
            if bools[i]:

                p1[i], p2[i] = p2[i], p1[i]

        if np.random.uniform() < 0.5:

            offspring_mutation_rates += [p1_sigma, p2_sigma]

        else:

            offspring_mutation_rates += [p2_sigma, p1_sigma]


        offspring += [p1, p2]


    return offspring, offspring_mutation_rates

# tournament
def tournament(population, population_fitness):
    
    c1_index = np.random.randint(population_size)
    c2_index = np.random.randint(population_size)

    if population_fitness[c1_index] > population_fitness[c2_index]:
        return c1_index
    else:
        return c2_index


def main():

    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'round_robin_selection'
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

    generations_max = 50
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

        #mean of the population performance
        mean = np.mean(population_fitness)

        #std of the population performance
        std = np.std(population_fitness)

        #index of the best_solution
        best_solution_index = np.argmax(population_fitness)

        #best performing individual
        best_chromosome = population[best_solution_index]

        #fitness of this individual
        best_fitness = population_fitness[best_solution_index]
        
        #mutation_rate of this individual
        best_mutation_rate = mutation_step_sizes[best_solution_index]
        
        save_results(experiment_name=experiment_name, ini_g=generation, best=best_fitness, mean=mean, std=std)

        #generate offspring by crossover
        offspring, offspring_step_sizes = crossover(population, mutation_step_sizes, population_fitness)

        #mutate offspring
        offspring, offspring_step_sizes = mutate_population(offspring, learning_rate_overall=learning_rate_overall, 
                           learning_rate_coordinate=learning_rate_coordinate, mutation_step_sizes=offspring_step_sizes)

        #evaluate new solutions
        offspring_fitness = evaluate(env, offspring)

        population = np.vstack((population, offspring))

        #cumulative fitness
        population_fitness = np.concatenate((population_fitness, offspring_fitness))

        #cumulative mutation step_sizes
        mutation_step_sizes = np.concatenate((mutation_step_sizes,offspring_step_sizes))

        population, mutation_step_sizes, population_fitness = round_robin(population, mutation_step_sizes, population_fitness)

        generation += 1

        
        
generations_max = 50
last_best = 0
lowerbound = -1
upperbound = 1
population_size = 100
learning_rate_overall = 1/((2*population_size)**0.5)-0.03
learning_rate_coordinate = 1/((2*(population_size**0.5))**0.5)-0.1
MUTATION_PROBABILITY = 0.8




if __name__ == '__main__':
    main()