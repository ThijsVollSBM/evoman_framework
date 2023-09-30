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

# update mutation rate (explorative) : uniform sampling from [min(x), max(x)] - Page 57
# There is no sigma for this kind of sampling.
# We can later add some "step size" just to make sure there are new values being introduced 
# to the gene pool each time and then


def initialize_population(env, experiment_name, lowerbound, upperbound,
                            population_size, n_vars):

    if not os.path.exists(experiment_name+'/evoman_explorative'):

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

def update_mutations_uniformly(population):

    return np.random.uniform(np.vstack(population).min(),
                              np.vstack(population).max(),
                              size = (100,265))

def mutate_population(some_population):

    mutated_offspring = update_mutations_uniformly(population=some_population)
    for i in range(len(mutated_offspring)):

        mutation_prob = np.random.uniform()

        if mutation_prob < MUTATION_PROBABILITY:

            some_population[i] = mutated_offspring[i]

    return some_population

def crossover(population, new_step_size_prob):

    offspring = []

    for i in range(0,population_size, 2):

        p1_index = np.random.randint(population_size)
        p2_index = np.random.randint(population_size)

        p1 = population[p1_index].copy()

        p2 = population[p2_index].copy()

        bools1 = np.random.choice(a=[False, True], size=p1.shape)
        bools2 = np.random.choice(a=[False, True], size=p1.shape)
        
        for i in range(len(bools1)):
            #if True, swap the genome of the two parents
            if bools1[i]:
                p1[i], p2[i] = p2[i], p1[i]

        new_solution = [p1, p2]

        for gene in range(len(new_solution)):
            if np.random.uniform() < new_step_size_prob:
                new_solution[gene] += np.random.normal(0,1)

        offspring += new_solution

        for j in range(len(bools2)):
            if bools2[j]:
                p1[j], p2[j] = p2[j], p1[j]

        new_solution = [p1, p2]

        for gene in range(len(new_solution)):
            if np.random.uniform() < new_step_size_prob:
                new_solution[gene] += np.random.normal(0,1)

        offspring += new_solution

    return offspring

def survival_selection(offspring, offspring_fitness): 
    #age_based: kill all parents, and keep only children.. effectively return crossover() values
    indecies = np.argsort(offspring_fitness)
    offspring = offspring[indecies[-100:]]
    offspring_fitness = offspring_fitness[indecies[-100:]]
    return offspring, offspring_fitness


def main():

    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'evoman_explorative'
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
        #fitness of this individual
        best_fitness = population_fitness[best_solution_index]
        
        #mutation_rate of this individual
        #best_mutation_rate = mutation_step_sizes[best_solution_index]
        
        save_results(experiment_name=experiment_name, ini_g=generation, best=best_fitness, mean=mean, std=std)

        #generate offspring by crossover
        offspring = crossover(population, new_step_size_prob)

        #mutate offspring
        offspring = mutate_population(offspring)
        offspring = np.vstack(offspring)
        
        #survival selection
        offspring_fitness = evaluate(env, offspring)
        offspring, offspring_fitness = survival_selection(offspring, offspring_fitness)

        #Just follow the naming conventions of the exploit code
        population = offspring
        population_fitness = offspring_fitness

        generation += 1

        
        
generations_max = 50
new_step_size_prob = 0.5
last_best = 0
lowerbound = -1
upperbound = 1
population_size = 100
MUTATION_PROBABILITY = 0.8




if __name__ == '__main__':
    main()