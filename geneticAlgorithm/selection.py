import random


def roulette_wheel_selection(offspring_arrays, fitness_values):
    total_fitness = sum(fitness_values)
    selection_prob = [f / total_fitness for f in fitness_values]
    r = random.random()
    cumulative_prob = 0
    for i, prob in enumerate(selection_prob):
        cumulative_prob += prob
        if r <= cumulative_prob:
            return offspring_arrays[i]


def tournament_selection(offspring_arrays, fitness_values, k=3):
    # Conduct a single tournament
    tournament = random.sample(list(zip(offspring_arrays, fitness_values)), k)
    winner = max(tournament, key=lambda x: x[1])
    return winner[0]


def rank_based_selection(offspring_arrays, fitness_values):
    # Rank the offspring based on their fitness
    sorted_indices = sorted(range(len(fitness_values)), key=lambda k: fitness_values[k], reverse=True)
    total_ranks = sum(range(1, len(offspring_arrays) + 1))
    rank_prob = [i / total_ranks for i in range(1, len(offspring_arrays) + 1)]
    # Perform a single selection
    r = random.random()
    cumulative_prob = 0
    for i, prob in enumerate(rank_prob):
        cumulative_prob += prob
        if r <= cumulative_prob:
            return offspring_arrays[sorted_indices[i]]


def best_fitness_selection(offspring_arrays, fitness_values):
    # Find the index of the offspring with the highest fitness
    best_idx = fitness_values.index(max(fitness_values))
    return offspring_arrays[best_idx]


def selection(offspring_arrays, fitness_values, selection_function):
    # Apply the provided selection function
    selected_offspring = selection_function(offspring_arrays, fitness_values)
    return selected_offspring
