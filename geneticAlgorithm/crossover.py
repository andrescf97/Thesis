import random
import numpy as np


def one_point_crossover(parent1, parent2, bounds):
    # Choose whether to crossover x (0) or y (1)
    crossover_point = random.randint(0, 1)

    # Swap the coordinates
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()
    offspring1[crossover_point], offspring2[crossover_point] = offspring2[crossover_point], offspring1[crossover_point]

    # Apply bounds
    offspring1 = np.clip(offspring1, bounds[0][0], bounds[1][1])
    offspring2 = np.clip(offspring2, bounds[0][0], bounds[1][1])

    return offspring1, offspring2


def two_point_crossover(parent1, parent2, bounds):
    # Swap both coordinates
    offspring1 = np.array([parent2[0], parent1[1]])
    offspring2 = np.array([parent1[0], parent2[1]])

    # Apply bounds
    offspring1 = np.clip(offspring1, bounds[0][0], bounds[1][1])
    offspring2 = np.clip(offspring2, bounds[0][0], bounds[1][1])

    return offspring1, offspring2


def uniform_crossover(parent1, parent2, bounds, crossover_rate):
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    for i in range(2):  # For both x and y coordinates
        if random.random() < crossover_rate:
            offspring1[i], offspring2[i] = offspring2[i], offspring1[i]

    # Apply bounds
    offspring1 = np.clip(offspring1, bounds[0][0], bounds[1][1])
    offspring2 = np.clip(offspring2, bounds[0][0], bounds[1][1])

    return offspring1, offspring2


def apply_crossover(mutated_arrays, crossover_func, bounds, crossover_rate):
    offspring_arrays = []

    for array in mutated_arrays:
        # Make a copy to avoid modifying the original array
        new_array = array.copy()

        # Determine the number of crossovers to perform
        num_crossovers = round(crossover_rate*len(array))

        for _ in range(num_crossovers):
            # Randomly select two chargers for crossover
            idx1, idx2 = random.sample(range(len(array)), 2)
            parent1 = new_array[idx1]
            parent2 = new_array[idx2]

            # Apply crossover function
            offspring1, offspring2 = crossover_func(parent1, parent2, bounds, crossover_rate)

            # Replace original chargers with offspring
            new_array[idx1], new_array[idx2] = offspring1, offspring2

        offspring_arrays.append(new_array)

    return offspring_arrays
