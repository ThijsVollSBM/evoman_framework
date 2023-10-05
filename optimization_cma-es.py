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

    step_coordinate_LR = coordinate_LR*np.random.normal(0,1,(len(mutation_rates),1))

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

    mutants = np.add(population,offspring_mutation_rates*np.random.normal(0,1,(len(population),1)))

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

def point_crossover(parents, mutation_step_sizes, population_fitness, mode='random', n_points=None):

    if mode == 'random':

        return random_crossover(parents, mutation_step_sizes, population_fitness)

    assert n_points is not None

    offspring = []

    offspring_mutation_rates = []

    for i in range(0,len(parents), 2):

        p1 = parents[i].copy()
        p1_sigma = mutation_step_sizes[i]

        p2 = parents[i+1].copy()
        p2_sigma = mutation_step_sizes[i+1]

        for j in range(10):

            c1 = p1.copy()
            c2 = p2.copy()

            c1_sigma = p1_sigma.copy()
            c2_sigma = p2_sigma.copy()

            c1_chunks = np.reshape(c1, (-1, n_points))
            c2_chunks = np.reshape(c2, (-1, n_points))

            #perform the crossover based on a boolean map
            for genome_set in range(0, n_points, 2):
                
                c1_chunks[genome_set], c2_chunks[genome_set] = c2_chunks[genome_set], c1_chunks[genome_set]


            if np.random.uniform() < 0.5:

                offspring_mutation_rates += [c1_sigma, c2_sigma]

            else:

                offspring_mutation_rates += [c2_sigma, c1_sigma]


            offspring += [c1_chunks.flatten(), c2_chunks.flatten()]


    return offspring, offspring_mutation_rates

def random_crossover(parents, mutation_step_sizes, population_fitness):

    offspring = []

    offspring_mutation_rates = []

    for i in range(0,len(parents), 2):

        p1 = parents[i].copy()
        p1_sigma = mutation_step_sizes[i]

        p2 = parents[i+1].copy()
        p2_sigma = mutation_step_sizes[i+1]

        for j in range(10):

            bools = np.random.choice(a=[False, True], size=p1.shape)

            c1 = p1.copy()
            c2 = p2.copy()

            c1_sigma = p1_sigma.copy()
            c2_sigma = p2_sigma.copy()

            #perform the crossover based on a boolean map
            for genome in range(len(bools)):
                
                #if True, swap the genome of the two parents
                if bools[genome]:

                    c1[genome], c2[genome] = c2[genome], c1[genome]


            if np.random.uniform() < 0.5:

                offspring_mutation_rates += [c1_sigma, c2_sigma]

            else:

                offspring_mutation_rates += [c2_sigma, c1_sigma]


            offspring += [c1, c2]


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
                    enemies=[7,8],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    multiplemode='yes',
                    level=2,
                    speed="fastest",
                    visuals=False)

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5


    # start writing your own code from here

    #initialize population
    population, population_fitness, mutation_step_sizes, generation = initialize_population(env=env, 
                                                                       experiment_name=experiment_name,
                                                                       lowerbound=lowerbound,
                                                                       upperbound=upperbound,
                                                                       population_size=population_size,
                                                                       n_vars=n_vars
                                                                       )

    #########################################
    #    User defined input parameters      #
    #########################################

    xmean = np.random.normal(size=(n_vars))
    sigma = 0.5



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

    """B = eye(N); % B defines the coordinate system
        39 D = eye(N); % diagonal matrix D defines the scaling
        40 C = B*D*(B*D)’; % covariance matrix
        41 eigeneval = 0; % B and D updated at counteval == 0
        42 chiN=Nˆ0.5*(1-1/(4*N)+1/(21*Nˆ2)); % expectation of
        """

    cc = (4+mu_eff/n_vars / (n_vars+4 + 2*mu_eff/n_vars))

    cs = (mu_eff+2) / (n_vars+mu_eff+5)

    c1 = 2 / (((n_vars+1.3)**2)+mu_eff)

    cmu = min(1-c1, (2*(mu_eff-2+(1/mu_eff)) / ((((n_vars+2)**2)+2*mu_eff)/2))); # for rank-mu update

    pc = np.zeros((n_vars,)) 
    ps = np.zeros((n_vars,)) 

    B_eye = np.identity(n_vars)
    D_eye = np.identity(n_vars)

    C = B_eye*D_eye*(B_eye*D_eye).transpose()

    eigeneval = 0

    chiN = (n_vars**0.5)*(1-(1/(4*n_vars))+(1/(21*n_vars**2)))

    print(chiN)

    offspring = np.zeros((offspring_size, n_vars))

    #generate and evaluate lambda amount of offspring:

    arz = np.random.normal(size=(offspring_size,n_vars))

    arx = np.zeros(arz.shape)

    for i in range(offspring_size):

        arx[i] = xmean + sigma*(B_eye @ D_eye @ arz[i])

    offspring_fitness = evaluate(env, arx)


    #sort offspring, select mu amount of children, and recompute xmean and zmean

    sorted_indices = np.argsort(-(offspring_fitness))
    arx = arx[sorted_indices]
    arz = arz[sorted_indices]

    xmean = np.dot(weights, arx[:mu])
    zmean = np.dot(weights, arz[:mu])

    ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff) * (B_eye @ zmean)
    hsig = (np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(counteval/population_size))) / chiN < 1.4+2/(n_vars+1)

    pc = (1-cc)*pc + hsig * np.sqrt(cc*(2-cc)*mu_eff) * (B_eye@D_eye@zmean)

    C = (1-c1-cmu) * C + c1 * (pc*pc.transpose() + (1-hsig) * cc*(2-cc) * C) + cmu * (B_eye@D_eye@arz[:mu].transpose()) @ np.diag(weights) @ (B_eye@D_eye@arz[:mu].transpose()).T
    

    """
    % Adapt covariance matrix C
    69 C = (1-c1-cmu) * C ... % regard old matrix % Eq. 47
    70 + c1 * (pc*pc’ ... % plus rank one update
    71 + (1-hsig) * cc*(2-cc) * C) ... % minor correction
    72 + cmu ... % plus rank mu update
    73 * (B*D*arz(:,arindex(1:mu))) ...
    74 * diag(weights) * (B*D*arz(:,arindex(1:mu)))’;
    """

generations_max = 50
last_best = 0
lowerbound = -1
upperbound = 1
population_size = 300





if __name__ == '__main__':
    main()