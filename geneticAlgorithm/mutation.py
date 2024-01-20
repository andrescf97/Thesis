import numpy as np
import random


def gaussian_mutation_2d(array, mutation_rate, mean=0, std_dev=1.0, bounds=None):
    # Round std_dev to the nearest integer
    std_dev = round(std_dev)

    for idx in range(len(array)):
        if random.random() > 0.5:  # Randomly decide whether to mutate this charger
            # Mutate x coordinate
            new_x = array[idx, 0] + np.random.normal(mean, std_dev)
            array[idx, 0] = np.clip(new_x, bounds[0][0], bounds[0][1])

            # Mutate y coordinate
            new_y = array[idx, 1] + np.random.normal(mean, std_dev)
            array[idx, 1] = np.clip(new_y, bounds[1][0], bounds[1][1])
    return array



def bit_flip_mutation_2d(array, mutation_rate, bounds):
    for idx in range(len(array)):
        if random.random() < mutation_rate:
            # Flip x coordinate
            array[idx, 0] = random.uniform(bounds[0][0], bounds[0][1])

            # Flip y coordinate
            array[idx, 1] = random.uniform(bounds[1][0], bounds[1][1])
    return array


def swap_mutation_2d_single_coord(array):
    # Randomly select two chargers
    idx1, idx2 = random.sample(range(len(array)), 2)

    # Randomly decide whether to swap x (0) or y (1) coordinate
    coord_to_swap = random.randint(0, 1)

    # Swap the selected coordinate
    array[idx1, coord_to_swap], array[idx2, coord_to_swap] = array[idx2, coord_to_swap], array[idx1, coord_to_swap]

    return array


def scramble_mutation_2d_single_coord(array):
    # Randomly decide whether to scramble x (0) or y (1) coordinate
    coord_to_scramble = random.randint(0, 1)

    # Select a random segment of the array to scramble
    start, end = sorted(random.sample(range(len(array)), 2))
    segment = array[start:end, coord_to_scramble].copy()

    # Scramble the selected segment
    np.random.shuffle(segment)

    # Replace the original segment with the scrambled one
    array[start:end, coord_to_scramble] = segment

    return array


def create_mutated_arrays(original_array, num_arrays, bounds, mutation_type, mean = 0, std = 1.0, mutation_rate=0.1):
    mutated_arrays = []

    # Define the mutation functions
    mutation_functions = {
        'gaussian': lambda arr: gaussian_mutation_2d(arr, mutation_rate=mutation_rate, mean=mean, std_dev=std, bounds=bounds),
        'bit_flip': lambda arr: bit_flip_mutation_2d(arr, mutation_rate, bounds=bounds),
        'swap': swap_mutation_2d_single_coord,
        'scramble': scramble_mutation_2d_single_coord
    }

    # Check if the mutation type is valid
    if mutation_type not in mutation_functions:
        raise ValueError(f"Unknown mutation type: {mutation_type}")

    # Apply the selected mutation function num_arrays times
    for _ in range(num_arrays):
        new_array = np.copy(original_array)
        mutated_array = mutation_functions[mutation_type](new_array)
        mutated_arrays.append(mutated_array)

    return mutated_arrays
