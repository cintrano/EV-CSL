import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from operators import crossovers as cross, mutations as mut
import problem
import numpy as np
from arguments import args


def add_individual(toolbox):
    # generation functions
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox.register("attr_int", np.random.randint, 0, 3)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=args.L)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def add_selection(toolbox, mode):
    if True:
        toolbox.register("select", tools.selTournament, tournsize=3)


def add_crossover(toolbox, mode):
    if mode == 'MERGING':
        toolbox.register("mate", cross.merging)
    elif mode == '1POINT':
        toolbox.register("mate", tools.cxOnePoint)
    elif mode == '2POINT':
        toolbox.register("mate", tools.cxTwoPoint)
    elif mode == 'CUPCAP':
        toolbox.register("mate", cross.cupcap)
    else:
        toolbox.register("mate", cross.cross_none)


def add_mutation(toolbox, mode):
    toolbox.register("mutate_custom", mut.custom)
    if mode == 'SHAKING':
        toolbox.register("mutate", mut.shaking)
    elif mode == 'NRAND':
        toolbox.register("mutate", mut.nrand)
    elif mode == 'UNIFORM':
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=3, indpb=0.5)
    else:
        toolbox.register("mutate", mut.mut_none)


def add_replacement(toolbox, mode):
    if mode == 'mu_lambda':
        toolbox.register("replacement", algorithms.eaMuCommaLambda)
    elif mode == 'mu_plus_lambda':
        toolbox.register("replacement", algorithms.eaMuPlusLambda)


def configure_ga(toolbox, sel, cross, mut, repl):
    add_selection(toolbox, sel)
    add_crossover(toolbox, cross)
    add_mutation(toolbox, mut)
    add_replacement(toolbox, repl)


# Press the green button in the gutter to run the script.
def execute(toolbox, pop_size, fitness, cxpb=0.5, mutpb=0.2, max_iter=100):
    pop = toolbox.population(n=pop_size)
    toolbox.register("evaluate", fitness)
    print(".")
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print(".")
    for ind in pop:
        toolbox.repair(ind)
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    # CXPB, MUTPB = 0.5, 0.2
    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]
    # Variable keeping track of the number of generations
    g = 0

    print(".")
    print(max(fits))
    # Begin the evolution
    while g < max_iter:  #max(fits) < 100 and g < max_iter:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                #toolbox.mutate(mutant)
                toolbox.mutate_custom(mutant)
                toolbox.repair(mutant)
                del mutant.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = toolbox.select(pop + offspring, k=pop_size)
        #pop[:] = offspring
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        if g % 10 == 0:
            print("        Sol size %s" % len(pop[0]))
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5

            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
    return fits


def run(fitness, pop_size, sel, cross, mut, repl, cross_prob=0.5, mut_prob=0.2, max_iter=1000):
    toolbox = base.Toolbox()
    toolbox.register("repair", problem.repair, fitness=fitness)
    add_individual(toolbox)
    configure_ga(toolbox, sel, cross, mut, repl)
    execute(toolbox, pop_size, fitness, cross_prob, mut_prob, max_iter)
