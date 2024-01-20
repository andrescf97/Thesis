from models import Graph_Neural_Network_subgraph
from geneticAlgorithm.mutation import create_mutated_arrays
from geneticAlgorithm import crossover, selection
from util import create_training_data, get_data_directories
import pandas as pd
import numpy as np


def create_new_charger_location_distribution(data_directories, charger_locations, gnn, num_arrays=10,
                                             mutation_type='gaussian', mutation_rate=0.1, std=1,
                                             crossover_type=crossover.uniform_crossover, crossover_rate=0.5,
                                             selection_type=selection.best_fitness_selection):
    # Define the bounds
    bounds = [(4456269.0, 4478126.2), (5326682.0, 5341867.8)]
    # Start of GA
    # Performing mutation
    mutated_arrays = create_mutated_arrays(charger_locations, num_arrays, bounds, mutation_type, 0, std, mutation_rate)
    # Performing crossover
    offsprings = crossover.apply_crossover(mutated_arrays, crossover_type, bounds, crossover_rate)
    # Initialize fitness_values array
    fitness_value = []
    # Create instance data for the model
    idata = create_training_data(data_directories)
    charger = data_directories["charger_data"]
    # Iterate through each 2d array in the offspring
    for idx, offspring in enumerate(offsprings):
        # Update the charger data in data_directories for this offspring
        charger["charger_x"] = offspring[:, 0]
        charger["charger_y"] = offspring[:, 1]
        # Create instance data for the model
        idata = create_training_data(data_directories)
        # Predict fitness (utilization) and store it
        fitness = gnn.predict(idata)
        fitness_value.append(fitness)
    # Perform selection
    charger_locations = selection.selection(offspring_arrays=offsprings, fitness_values=fitness_value,
                                            selection_function=selection_type)
    # Update charger locations after selection
    charger["charger_x"], charger["charger_y"] = charger_locations[:, 0], charger_locations[:, 1]
    return charger


if __name__ == '__main__':
    # Defining number of generations.
    num_generations = 35
    initial_std = 5  # Starting value for std
    decay_rate = 0.95  # Decay rate per generation for std
    initial_mutation_rate = 0.5  # Starting value for mutation rate
    final_mutation_rate = 0.01  # Final value for mutation rate
    mutation_rate_decay = (initial_mutation_rate - final_mutation_rate) / num_generations
    num_iterations = 1
    prediction = 0
    fitness_values = []
    model = Graph_Neural_Network_subgraph()
    model.load()
    data_directory = get_data_directories()
    chargers = pd.read_csv(data_directory["charger_file"], sep=";")
    data_directory["charger_data"] = chargers
    x_locations = chargers["charger_x"].values
    y_locations = chargers["charger_y"].values
    # Combining into a single Numpy array
    charger_loc = np.column_stack((x_locations, y_locations))
    cache_charger_locations = [np.zeros((386, 386)) for _ in range(100)]
    for generation in range(num_generations):
        current_std = initial_std * (decay_rate ** generation)
        current_mutation_rate = max(initial_mutation_rate - (generation * mutation_rate_decay), final_mutation_rate)
        charger_location_distribution = create_new_charger_location_distribution(
            data_directory, charger_loc, model, num_arrays=100, mutation_type='gaussian',
            std=current_std, mutation_rate=current_mutation_rate, crossover_type=crossover.uniform_crossover,
            crossover_rate=current_mutation_rate, selection_type=selection.best_fitness_selection)
        data_directory["charger_data"] = charger_location_distribution
        x = charger_location_distribution["charger_x"].values
        y = charger_location_distribution["charger_y"].values
        # Combining into a single Numpy array
        charger_log = np.column_stack((x, y))
        # Save each charger location distribution to select the best performing one
        cache_charger_locations[generation] = charger_log
        # Predict final fitness using the GNN model
        instance_data = create_training_data(data_directory)
        prediction = model.predict(instance_data)
        fitness_values.append(prediction)  # Store the fitness value

