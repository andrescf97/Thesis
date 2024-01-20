import matplotlib.pyplot as plt


def plot_generation_vs_fitness(generations, fitness_values):
    """
    Plots the genetic algorithm's fitness values across generations.

    Parameters:
    generations (list): A list of generation numbers.
    fitness_values (list): A list of fitness values corresponding to each generation.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitness_values, marker='o', linestyle='-')
    plt.title('Generation vs. Fitness')
    plt.xlabel('Generation Number')
    plt.ylabel('Fitness Value')
    plt.grid(True)
    plt.show()


def plot_generation_vs_fitness_comparisons(generations, fitness_values1, fitness_values2, fitness_values3, labels):
    """
    Plots the genetic algorithm's fitness values across generations for three different mutation types.

    Parameters:
    generations (list): A list of generation numbers.
    fitness_values1, fitness_values2, fitness_values3 (list): Lists of fitness values for each mutation type.
    labels (list): List of labels for the mutation types.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(generations, fitness_values1, marker='o', linestyle='-', label=labels[0])
    plt.plot(generations, fitness_values2, marker='o', linestyle='-', label=labels[1])
    plt.plot(generations, fitness_values3, marker='o', linestyle='-', label=labels[2])

    plt.title('Generation vs. Fitness for Different Mutation Rates')
    plt.xlabel('Generation Number')
    plt.ylabel('Fitness Value')
    plt.legend()
    plt.grid(True)
    plt.show()
