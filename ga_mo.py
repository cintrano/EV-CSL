
from deap import base
from deap import creator
from deap import tools
from operators import crossovers as cross, mutations as mut
import main
import problem
import numpy as np

from arguments import args


def configure_individual(toolbox):
    # generation functions
    creator.create("FitnessMaxMin", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMaxMin)

    toolbox.register("attr_int", np.random.randint, 0, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=args.L)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def configure_selection(toolbox, mode):
    if mode == 'TOURNAMENT2':
        toolbox.register("select", tools.selTournament, tournsize=2)
    elif mode == 'TOURNAMENT3':
        toolbox.register("select", tools.selTournament, tournsize=3)
    elif mode == 'RANDOM':
        toolbox.register("select", tools.selRandom)
    elif mode == 'BEST':
        toolbox.register("select", tools.selBest)
    elif mode == 'WORST':
        toolbox.register("select", tools.selWorst)
    elif mode == 'ROULETTE':
        toolbox.register("select", tools.selRoulette)
    elif mode == 'DOUBLE_TOURNAMENT':
        pass
    elif mode == 'STOCHASTIC_UNIVERSAL_SAMPLING':
        toolbox.register("select", tools.selStochasticUniversalSampling)
    elif mode == 'LEXICASE':
        toolbox.register("select", tools.selLexicase)
    elif mode == 'EPSILON':
        pass
    elif mode == 'AUTOMATIC_EPSILON_LEXICASE':
        toolbox.register("select", tools.selAutomaticEpsilonLexicase)


def configure_crossover(toolbox, mode):
    if mode == 'MERGING':
        toolbox.register("mate", cross.merging)
    elif mode == '1POINT':
        toolbox.register("mate", tools.cxOnePoint)
    elif mode == '2POINT':
        toolbox.register("mate", tools.cxTwoPoint)
    elif mode == 'CUPCAP':
        toolbox.register("mate", cross.cupcap)
    elif mode == 'ZONES':
        toolbox.register("mate", cross.cross_by_substations_zone)
    else:
        toolbox.register("mate", cross.cross_none)


def configure_mutation(toolbox, mode):
    toolbox.register("mutate_custom", mut.custom)
    if mode == 'SHAKING':
        toolbox.register("mutate", mut.shaking)
    elif mode == 'NRAND':
        toolbox.register("mutate", mut.nrand)
    elif mode == 'UNIFORM':
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=2, indpb=0.5)
    else:
        toolbox.register("mutate", mut.mut_none)


def configure_replacement(toolbox, mode):
    # if mode == 'mu_lambda':
    #     toolbox.register("replacement", algorithms.eaMuCommaLambda)
    # elif mode == 'mu_plus_lambda':
    #     toolbox.register("replacement", algorithms.eaMuPlusLambda)
    pass


def configure_ga(toolbox, sel, cross, mut, repl):
    configure_selection(toolbox, sel)
    configure_crossover(toolbox, cross)
    configure_mutation(toolbox, mut)
    configure_replacement(toolbox, repl)


def print_stats(fits):
    length = len(fits)
    mean_1 = sum(i for i, _ in fits) / length
    mean_2 = sum(i for _, i in fits) / length
    sum2_1 = sum(x[0] * x[0] for x in fits)
    sum2_2 = sum(x[1] * x[1] for x in fits)
    std_1 = abs(sum2_1 / length - mean_1 ** 2) ** 0.5
    std_2 = abs(sum2_2 / length - mean_2 ** 2) ** 0.5
    print("  Min (%s, %s)" % (min(i for i, _ in fits), min(i for _, i in fits)))
    print("  Max (%s, %s)" % (max(i for i, _ in fits), max(i for _, i in fits)))
    print("  Avg (%s, %s)" % (mean_1, mean_2))
    print("  Std (%s, %s)" % (std_1, std_2))


# Press the green button in the gutter to run the script.
def execute(toolbox, pop_size, fitness, cxpb=0.5, mutpb=0.2, max_iter=100):
    """GA algorithm
    :param toolbox: Deap toolbox instance
    :param pop_size: Size of the population
    :param fitness: Fitness function
    :param cxpb: The probability with which two individuals are crossed
    :param mutpb: The probability for mutating an individual
    :param max_iter: Number of iterations of the algorithm
    :return:
    """
    pop = toolbox.population(n=pop_size)
    toolbox.register("evaluate", fitness)
    # Repair the initial population
    for ind in pop:
        toolbox.repair(ind)
    # Evaluate the entire population
    # fitnesses = list(map(toolbox.evaluate, pop))
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    # Extracting all the fitnesses of
    fits = [ind.fitness.values for ind in pop]
    # Variable keeping track of the number of generations
    g = 0
    # Begin the evolution
    while g < max_iter:  #max(fits) < 100 and g < max_iter:
        # A new generation
        g = g + 1
        # if g % 50 == 0:
        # operators    print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < mutpb:
                # toolbox.mutate(mutant)
                toolbox.mutate_custom(mutant)
                toolbox.repair(mutant)
                del mutant.fitness.values
        # Evaluate the individuals
        fitnesses = map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        pop[:] = toolbox.replacement(pop + offspring)
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values for ind in pop]
        if g % 10 == 0:
            # print_stats(fits)
            # main.write_fitness_values(fits, f"fits-{g}.ssv")
            # main.write_population(pop, f"pops-{g}.ssv")
            pass
    return fits, pop


toolbox = base.Toolbox()


def run_nsga2(fitness, pop_size, sel, cross, mut, repl, cross_prob=0.5, mut_prob=0.2, max_iter=1000):
    global toolbox
    toolbox.register("repair", problem.repair)
    toolbox.register("replacement", tools.selNSGA2, k=pop_size)
    configure_individual(toolbox)
    configure_ga(toolbox, sel, cross, mut, repl)
    return execute(toolbox, pop_size, fitness, cross_prob, mut_prob, max_iter)


def run_spea2(fitness, pop_size, sel, cross, mut, repl, cross_prob=0.5, mut_prob=0.2, max_iter=1000):
    global toolbox
    toolbox.register("repair", problem.repair)
    toolbox.register("replacement", tools.selSPEA2, k=pop_size)
    configure_individual(toolbox)
    configure_ga(toolbox, sel, cross, mut, repl)
    return execute(toolbox, pop_size, fitness, cross_prob, mut_prob, max_iter)
