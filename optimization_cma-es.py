###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

import time

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os
from numpy import linalg as LA

def cons_multi_patch(values):

        return values


def simulation(env,x, evaluate_best=False):

    global enemy_weights

    f,p,e,t = env.play(pcont=x)

    if not evaluate_best:
        return f, e
    
    else:

        print(80*'*')
        print('\n')
        print('player life:')
        print(p)
        print('enemy life:')
        print(e)
        print('\n')
        print(f'vanilla player fitness:\t {f.mean() - f.std()}')
    

def calculate_weighted_fitness(vector):

    global enemy_weights

    weighted_fitness =  vector * (enemy_weights*len(enemy_weights))
    
    return np.average(weighted_fitness) - np.std(weighted_fitness)


def calculate_weighted_scores(vector):

    global enemy_weights

    scores = (10000 - np.square(vector)) * (enemy_weights*len(enemy_weights))

    avg_score = np.average(scores)

    kill_scores = np.count_nonzero(vector==0)*10000

    return avg_score+ kill_scores
    

def evaluate(env, x):

    global enemy_weights

    enemy_life_vectors = []

    fitness_vectors = []

    for gene in x:

        v_fitness, v_enemy_life = simulation(env, gene)

        #print(v_fitness.mean() - v_fitness.std())

        enemy_life_vectors.append(v_enemy_life)

        if v_enemy_life.sum() == 0:
            
            np.savetxt(f'all_enemies_beat_{time.strftime("%Y%m%d-%H%M%S")}.txt', gene)

        fitness_vectors.append(v_fitness)

    avg_enemy_life = np.vstack(enemy_life_vectors).mean(axis=0)
    
    norm_enemy_life = avg_enemy_life/avg_enemy_life.sum()

    wA = 0.6
    wB = 0.4

    enemy_weights = (enemy_weights * wA) + (norm_enemy_life * wB)

    scores = np.array(list(map(lambda y: calculate_weighted_scores(y), enemy_life_vectors)))

    if scores.sum() == 0:

        return np.array(list(map(lambda y: calculate_weighted_fitness(y), fitness_vectors)))
    
    else:

        return np.array(list(map(lambda y: calculate_weighted_scores(y), enemy_life_vectors)))


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


def dot_itself(matrix):
    
    matrix_list = []
    
    for vector in matrix:
        
        new_matrix = np.outer(vector, vector.T)
        matrix_list.append(new_matrix)
        
    return np.stack(matrix_list, axis = 0)


def main(enemies_list):

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
                    enemies=enemies_list,
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    multiplemode='yes',
                    level=2,
                    speed="fastest",
                    visuals=False)

    
    env.cons_multi = cons_multi_patch

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    # start writing your own code from here

    global enemy_weights

    #enemy_weights = np.ones(len(enemies_list))

    enemy_weights = np.array([0.5, 0, 0, 0.5, 0, 0, 0, 0])/len(enemies_list)

    #initialize population
    population, population_fitness, mutation_step_sizes, generation = initialize_population(env=env, 
                                                                       experiment_name=experiment_name,
                                                                       lowerbound=lowerbound,
                                                                       upperbound=upperbound,
                                                                       population_size=population_size,
                                                                       n_vars=n_vars
                                                                       )

    overall_best = 0

    #########################################
    #    User defined input parameters      #
    #########################################

    xmean = np.random.normal(size=(n_vars))
    sigma = 0.5
    stopeval = 1000000
    
    #########################################
    # Strategy parameter setting: Selection #
    #########################################
    
    #calculate lambda population size
    offspring_size = int(4 + np.floor(3*np.log(n_vars)))

    #set mu first
    mu = int(offspring_size/2)

    #calculate the weights
    weights = np.log(mu+1/2) - np.log(range(1,mu+1))

    #floor mu to get the number of parents
    mu = int(np.floor(mu))

    #normalize the weights
    weights = weights/sum(weights)

    #calculate the effective size of mu
    mu_eff = sum(weights)**2/sum(weights**2)


    ##########################################
    # Strategy parameter setting: Adaptation #
    ##########################################

    cc = (4+mu_eff/n_vars) / (n_vars+4 + 2*mu_eff/n_vars)

    cs = (mu_eff+2) / (n_vars+mu_eff+5)

    c1 = 2 / (((n_vars+1.3)**2)+mu_eff)

    cmu = min(1-c1, (2*(mu_eff-2+(1/mu_eff)) / (((n_vars+2)**2)+2*mu_eff/2))); # for rank-mu update

    damps = 1+2*max(0,np.sqrt((mu_eff -1)/(n_vars+1))-1) + cs

    ##########################################
    # initialize constants and strat params  #
    ##########################################

    pc = np.zeros((1,n_vars)) 
    
    ps = np.zeros((1,n_vars))

    B_eye = np.identity(n_vars)
    D_eye = np.identity(n_vars)

    C = B_eye * D_eye * (B_eye*D_eye).T

    eigeneval = 0

    chiN = (n_vars**0.5)*(1-(1/(4*n_vars))+(1/(21*n_vars**2)))

    counteval = 0
    
    generation = 0

    while counteval < stopeval:
        
        generation += 1

        #generate and evaluate lambda amount of offspring:
        arz = np.zeros((offspring_size,n_vars))       
        arx = np.zeros(arz.shape)

        for i in range(offspring_size):
            
            counteval += 1

            arz[i] = np.random.normal(0,1,size=(1,265))

            Yk = arz[i] @ (B_eye * D_eye)                        #eq. 39
            
            arx[i] = xmean + sigma * Yk                            #eq. 40
            
        offspring_fitness = evaluate(env, arx)
    
       

        #sort offspring, select mu amount of children, and recompute xmean and zmean
        sorted_indices = np.argsort(-offspring_fitness)
        arx = arx[sorted_indices]
        arz = arz[sorted_indices]
        offspring_fitness = offspring_fitness[sorted_indices]
  

        xmean = np.dot(weights, arx[:mu])
        zmean = np.dot(weights, arz[:mu])


        #update evolution paths
        ps = (1-cs)*ps + (np.sqrt(cs*(2-cs)*mu_eff)) * (B_eye @ zmean)  #eq. 43
        hsig = (np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*counteval/population_size))) / chiN < 1.4+2/(n_vars+1)

        pc = (1-cc)*pc + hsig * np.sqrt(cc*(2-cc)*mu_eff) * (B_eye * D_eye@zmean)

        element1 = (1-c1-cmu) * C   #can be close or equal to 0
        element2 = c1 * (pc*pc.T + (1-hsig) * cc*(2-cc) * C)   #rank-one update
        element3 = cmu*(np.sum(dot_itself(arz[:mu]) * weights[:, np.newaxis, np.newaxis], axis = 0))
        
        #element3 = cmu * ((B_eye * D_eye@arz[:mu].transpose()) * np.diag(weights) * (B_eye * D_eye@arz[:mu].transpose()).T)

        C = element1 + element2 + element3
        
        #adapt stepsize sigma
        #sigma = sigma * np.exp((cs/damps) * (np.linalg.norm(ps) / chiN - 1))

        #update B and D from C
        if counteval - eigeneval > population_size / (1 + cmu) / n_vars / 10:

            eigeneval = counteval
            C = np.tril(C) + np.triu(C.T, 1)
            D_eye, B_eye = LA.eigh(C)
            D_eye = np.diag(np.sqrt(np.diag(D_eye)))


        """in MatLab, the eig function returns eigenvectors, eigenvalues,  
        but numpy returns eigenvalues, eigenvectors so flipped the return statement"""

        population_fitness = offspring_fitness
        #mean of the population performance
        mean = np.mean(population_fitness)

        #std of the population performance
        std = np.std(population_fitness)

        #index of the best_solution
        best_solution_index = np.argmax(population_fitness)

        best_gene = arx[0]

        simulation(env,best_gene, evaluate_best = True)
    
        #fitness of this individual
        best_fitness = population_fitness[best_solution_index]

        if best_fitness > overall_best:
            overall_best = best_fitness
            np.savetxt('best_cma.txt', arx[0])

        print(f'overall_best: \t \t {overall_best}')
        save_results(experiment_name=experiment_name, ini_g=generation, best=best_fitness, mean=mean, std=std)

        """
        if generation % 10 == 0:
            print('enemy weights:')
            print(enemy_weights)
            """




lowerbound = -1
upperbound = 1
population_size = 20





if __name__ == '__main__':
    enemies = [1,2,3,4,5,6,7,8]
    main(enemies)