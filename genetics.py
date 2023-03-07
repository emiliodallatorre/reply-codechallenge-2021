from random import randint, random

from utils import get_score as get_fitness

MAX_LOOPS: int = 10000
MUTATION_RATE: float = 0.2


def crossover(parent_1: list, parent_2: list, meaningful_positions: list) -> list:
    crossover_point: int = randint(0, len(parent_1) - 1)

    child_1: list = parent_1[:crossover_point] + parent_2[crossover_point:]
    child_2: list = parent_2[:crossover_point] + parent_1[crossover_point:]

    if random() < MUTATION_RATE:
        child_1 = mutate(child_1, meaningful_positions)
        child_2 = mutate(child_2, meaningful_positions)

    return [parent_1, parent_2, child_1, child_2]


def mutate(individual: list, meaningful_positions: list) -> list:
    mutation_point: int = randint(0, len(individual) - 1)
    individual[mutation_point] = meaningful_positions[randint(0, len(meaningful_positions) - 1)]

    return individual


def do_genetics(buildings_positions: list, buildings_speed_score: list,
                buildings_latency_score: list, antennas_range: list, antennas_speeds: list, reward: int,
                meaningful_positions: list) -> tuple:
    chromosome_length: int = len(antennas_range)
    population: list = []

    last_individual: list = []
    for position in meaningful_positions:
        last_individual.append(position)

        if len(last_individual) == chromosome_length:
            population.append(last_individual)
            last_individual = []

    if len(last_individual) > 0:
        for i in range(chromosome_length - len(last_individual)):
            last_individual.append(meaningful_positions[i])

        population.append(last_individual)

    if len(population) % 2 != 0:
        population.append(population[0])

    best_fitnesses_by_generation: list = []
    for i in range(MAX_LOOPS):
        population.sort(key=lambda individual: get_fitness(buildings_positions, individual, buildings_speed_score,
                                                           buildings_latency_score, antennas_range, antennas_speeds,
                                                           reward), reverse=True)

        population = population[:len(population) // 2]

        best_fitnesses_by_generation.append(get_fitness(buildings_positions, population[0], buildings_speed_score,
                                                        buildings_latency_score, antennas_range, antennas_speeds,
                                                        reward))
        new_population: list = []

        for j in range(0, len(population), 2):
            new_population.extend(crossover(population[j], population[j + 1], meaningful_positions))

        population = new_population

    return population[0], (range(MAX_LOOPS), best_fitnesses_by_generation)
