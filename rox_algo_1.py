import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

import time
import numpy as np
import os


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

#TODO: Change for our own experiments
experiment_name = 'rox_algo_1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10


class ModifiedEnvironment(Environment):
    def fitness_single(self):
        return 0.9*(100 - self.get_enemylife()) + 0.1*self.get_playerlife() - np.log(self.get_time())

# initializes simulation in individual evolution mode, for single static enemy.
env = ModifiedEnvironment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5


dom_u = 1
dom_l = -1
npop = 100
gens = 30
mutation = 0.2
last_best = 0
tau = 1 / np.sqrt(npop)   # For Self-Adaptive Mutation


# runs simulation
def simulation(env,x):
    # Simulate a game by passing the population like that
    f,p,e,t = env.play(pcont=x)
    # It will directly return 100 fitness values for the simulation. We have to rewrite it from the Environment class!
    return f

# normalizes
def norm(x, pfit_pop):

    if ( max(pfit_pop) - min(pfit_pop) ) > 0:
        x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm


# evaluation
def evaluate(x):
    print("BEFORE evaluation: ", x)
    return np.array(list(map(lambda y: simulation(env,y), x)))


# tournament
def tournament(pop, k=2):
    
    # The probability p that the most fit member of the tournament is selected.
    # Usually this is 1 (deterministic tournaments), but stochastic versions are
    # also used with p < 1. Since this makes it more likely that a less-fit member
    # will be selected, decreasing p will decrease the selection pressure.
    # ALSO we applied the Tournament with replacement!
    
    b =  np.random.randint(0,pop.shape[0], k)
    max_index = np.where(fit_pop[b] == max(fit_pop[b]))
    print("HERE: ", b, fit_pop[b], max_index)

    return pop[max_index][0]


# limits
def limits(x):

    if x>dom_u:
        return dom_u
    elif x<dom_l:
        return dom_l
    else:
        return x


# crossover
def crossover(pop):

    total_offspring = np.zeros((0, n_vars))

    for _ in range(0, pop.shape[0], 2):   # For 100/2 = 50 pairs, make [1, 3] babies

        p1 = tournament(pop, k=int(npop/10))
        p2 = tournament(pop, k=int(npop/10))

        # n_offspring =   np.random.randint(1, 3+1, 1)[0]
        n_offspring = 2
        offspring =  np.zeros((n_offspring, n_vars))
        print("NUMBER OF OFFSRPING: ", n_offspring)

        for f in range(0, n_offspring):

            cross_prop = np.random.uniform(0,1)

            simple_crossover_index = np.random.randint(0, p1.shape[0], 1)[0]
            # print("Select random index: ", simple_crossover_index)
            print(p1[simple_crossover_index:].shape)
            # print("Cross prop: ", cross_prop)
            # print("Parent 1: ", p1)
            # print("Parent 2: ", p2)

            if f == 0:
                offspring[f][:simple_crossover_index] = p1[:simple_crossover_index]
            else:
                offspring[f][:simple_crossover_index] = p2[:simple_crossover_index]
            
            offspring[f][simple_crossover_index:] = p1[simple_crossover_index:] * cross_prop + p2[simple_crossover_index:] * (1 - cross_prop)
            # print("OFFSRPING: ", offspring[f])


            sigma_prime = 1 * np.exp(tau * np.random.normal(0, 1))
            # mutation
            for i in range(0, len(offspring[f])):
                if np.random.uniform(0, 1) <= mutation:
                    # print(offspring[f][i])
                    offspring[f][i] = offspring[f][i] + sigma_prime * np.random.normal(0, 1)
                    # print(offspring[f][i], '\n')

            offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring


# kills the worst genomes, and replace with new best/random solutions
def doomsday(pop,fit_pop):

    worst = int(npop/4)  # a quarter of the population
    order = np.argsort(fit_pop)
    orderasc = order[0:worst]

    for o in orderasc:
        for j in range(0,n_vars):
            pro = np.random.uniform(0,1)
            if np.random.uniform(0,1)  <= pro:
                pop[o][j] = np.random.uniform(dom_l, dom_u) # random dna, uniform dist.
            else:
                pop[o][j] = pop[order[-1:]][0][j] # dna from best

        fit_pop[o]=evaluate([pop[o]])

    return pop,fit_pop



# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)


# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))           # Pop size (100, 265)
    # print("POP: ", pop.shape)
    # print("np.random: ", dom_l, dom_u)
    fit_pop = evaluate(pop)
    # print("Fitness: ", fit_pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    initial_gen = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

else:

    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    # finds last generation number
    file_aux  = open(experiment_name+'/gen.txt','r')
    initial_gen = int(file_aux.readline())
    file_aux.close()




# saves results for first pop
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(initial_gen)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(initial_gen)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.close()


# evolution

last_sol = fit_pop[best]
notimproved = 0
print("Last_sol: ", last_sol) #Best fitness in the generation 0

for i in range(initial_gen+1, gens):

    offspring = crossover(pop)  # crossover
    
    
    print("OFFSRPING: ", offspring.shape)
    fit_offspring = evaluate(offspring)   # evaluation
    print("FITNESS FINAL: ", fit_offspring)
    pop = np.vstack((pop,offspring))
    fit_pop = np.append(fit_pop,fit_offspring)

    best = np.argmax(fit_pop) #best solution in generation
    fit_pop[best] = float(evaluate(np.array([pop[best] ]))[0]) # repeats best eval, for stability issues
    best_sol = fit_pop[best]

    # selection
    fit_pop_cp = fit_pop
    fit_pop_norm =  np.array(list(map(lambda y: norm(y,fit_pop_cp), fit_pop))) # avoiding negative probabilities, as fitness is ranges from negative numbers
    probs = (fit_pop_norm)/(fit_pop_norm).sum()
    chosen = np.random.choice(pop.shape[0], npop , p=probs, replace=False)
    print("Chosen: ", chosen)  #Which 100 out of 199 it chose!
    chosen = np.append(chosen[1:],best)
    pop = pop[chosen]                      #THE NEW POPULATION
    fit_pop = fit_pop[chosen]              #THE FITNESS OF THE NEW POPULATION


    # searching new areas

    if best_sol <= last_sol:
        notimproved += 1
    else:
        last_sol = best_sol
        notimproved = 0

    if notimproved >= 15:

        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\ndoomsday')
        file_aux.close()

        pop, fit_pop = doomsday(pop,fit_pop)
        notimproved = 0

    best = np.argmax(fit_pop)
    std  =  np.std(fit_pop)
    mean = np.mean(fit_pop)


    # saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    # saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name+'/best.txt',pop[best])

    # saves simulation state
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)
    env.save_state()




fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')


file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()


env.state_to_log() # checks environment state
