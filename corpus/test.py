"""
Genetic Algorithm

This script borrows the techniques from natural computation, using Genetic Algorithm to 
search for such optimial solutions through fitness function, in the application of 
music creativity. Like the original proposed solution of the genetic algorithm, 
the fitness function is evaluated by its sum of its chosen candidate, with the 
addition of subjectivity from the user to decide their choice of the best melodic 
composition composed by its application. The following script is utilized after the 
execution from the Dialog window where the user can display the evolution that is 
used from this script.

The following methods that were implemented for this script are the listed.

Methods
-------

insert_mode(mode):
    Selects the numerical value for the mode based on the string
    
insert_key(key):
    Selects the numerical value for the key.

generate_individual(note_quantity, mode, key, tempo):
    Produces a candidate for the population with user's preferences

selection(selection_method, num_to_replace, population, fitness_scores):
    Directs the type of selection method that will be used to select candidates during selection stage

calculate_fitness(individual):
    Calculcates the fitness function of the individual

roulette_wheel_selection(population, fitness_scores):
    Randomly selects candidates based on its selection probability 

rank_selection(population, fitness_scores):
    Ranks all the candidates from the population based on its fitness

stochastic_universal_sampling(population, fitness_scores): 
    Randomly selects candidates using evenly spaced intervals for weak candidates to be selected

tournament_selection(population, fitness_scores):
    Selects the fittest candidates from the current generation through K-way tournament

elitist_selection(population, fitness_scores):
    Selects limited number of candidates with the highest fitness values

crossover(parent1, parent2):
    Combines the genetic information of two parents to generate new offspring

insertion_mutation(individual, mutation_rate):
    Randomly selects a location of the gene and replaces the value

inversion_mutation(individual, mutation_rate):
    Randomly choses two points of the gene and inverts the values

scramble_mutation(individual, mutation_rate):
    Randomly selects two points of the gene and places the values randomly in the following two points

swap_mutation(individual, mutation_rate):
    Randomly selects two genes and swap its location for both genes

random_mutation_type(individual, mutation_rate):
    Randomly selects any type of mutation genetic operator for generating offspring

mutate(individual_type, individual, mutation_rate):
    Selects a mutation type for generating offspring based of user's preference

calculate_best_individual(fitness_scores):
    Calculates the candidate with the highest fitness value

best_individual_as_array(best_individual_per_generation, best_individual, max_fitness, fitness_scores, generation):
    Appends the candidate with the highest fitness value from the top five selection to an array per generation

sort_top_five_individuals(fitness_scores):
    Sorts the top five candidates based on its fitness scores

selecting_top_five_individual(individual, order, population, fitness_scores):
    Selects the candidates that have the highest fitness values on the top five

create_MIDI(export_directory, order, i, selected_individual, octave):
    Produces a MIDI track based on the user's preference

adding_individual_to_top_five(top_five_individuals,_indexes, individual_index, population, order):
    Appends the selected individual to the top five if the individual has the highest fitness value

find_index(arr, value):
    Finds the index of a given value

replacing_value_of_population_index(population, new_five_individuals):
    Replaces a value from a specified index if the following candidate equates to 0

print_sorted_highest_individual(best_individual_per_generation):
    Print the candidate that has the highest fitness value of all generation

evolution(export_directory, selection_method: str, generation: int, population: int, mutation_rate: float, mutation_type: str, octave: int, replace):
    Executes the evolution stage to generate new offsprings and measure the fitness value of each candidate per generation

generating_population(note_quantity, mode, key, tempo, population_size):
    Produces a population based on the user's preference
    
arguments():
    Passes the given arguments as values for preparing the generation of candidates
    
"""

import os 
import sys
import argparse

import random
import numpy as np
from functools import *

from music import *


def insert_mode(mode):
    """Selects the numerical value for the mode based on the string

    Args:
        mode (str): The musical scale that are defined by their starting note or tonic.

    Returns:
        mode (int): The numerical value for mode
    """
    match mode:
        case "Major":
            return 0
        case "Dorian":
            return 1
        case "Phrygian":
            return 2
        case "Lydian":
            return 3
        case "Mixolydian":
            return 4
        case "Minor":
            return 5
        case "Locrian":
            return 6
        case "Harmonic Minor":
            return 7
        case "Melodic Minor":
            return 8
        case "Neapolitan Minor":
            return 9
        case "Hungarian Minor":
            return 10
        case "Pentatonic Major":
            return 11
        case "Pentatonic Minor":
            return 12
        case "Blues":
            return 13
    print("Mode")
    
def insert_key(key):
    """Selects the numerical value for the key.

    Args:
        key (string): Group of pitches or scale that requires the basis of a musical composition

    Returns:
        key (int): The numerical value representation of the key
    """
    match key:
        case "C":
            return 0
        case "C# / Db":
            return 1
        case "D":
            return 2
        case "D# / Eb":
            return 3
        case "E":
            return 4
        case "F":
            return 5
        case "F# / Gb":
            return 6
        case "G":
            return 7
        case "G# / Ab":
            return 8 
        case "A":
            return 9
        case "A# / Bb":
            return 10
        case "B":
            return 11
        case "C":
            return 12
    print("Key")

def generate_individual(note_quantity, mode, key, tempo):
    """Produces a candidate for the population with user's preferences

    Args:
        note_quantity (int): The amount of notes that will be performed from a musical piece
        mode (str): The musical scale that are defined by their starting note or tonic
        key (int): Group of pitches or scale that requires the basis of a musical composition
        tempo (int): The speed or pace of a given piece it will be performed in

    Returns:
        individual (array): A candidate from a population in its generation
    """
    individual = [random.randint(0, 7) for _ in range(note_quantity)]
    individual[0] = insert_mode(mode)
    individual[1] = insert_key(key)
    individual[2] = tempo
    return individual

def selection(selection_method, num_to_replace, population, fitness_scores):
    """Directs the type of selection method that will be used to select candidates during selection stage

    Args:
        selection_method (str): The method of how candidates are chosen from a given population
        population (int): The number of candidates that will be converged in a generation
        fitness_scores (int): The fitness value of each candidate of a given population
        
    Returns:
        selection_method(population, fitness_scores): The type of selection method in its selection stage
    """
    match selection_method:
        case 'Roulette Wheel Selection':
            return roulette_wheel_selection(population, fitness_scores)
        case 'Rank Selection':
            return rank_selection(population, fitness_scores) 
        case 'Tournament Selection':
            return tournament_selection(population, fitness_scores)
        case 'Stochastic Universal Sampling':
            return stochastic_universal_sampling(population, fitness_scores) 
        case 'Elitist Selection':
            return elitist_selection(population, fitness_scores) 

def calculate_fitness(individual):
    """Calculates the fitness function of an individual

    Args:
        individual (_type_): A candidate from a population in its generation

    Returns:
        sum(individual): The sum of each numerical values from its candidate
    """
    return sum(individual)

def roulette_wheel_selection(population, fitness_scores):
    """Randomly selects candidates based on its selection probability 

    Args:
        population (int): The number of candidates that will be converged in a generation
        fitness_scores (array): The fitness value of each candidate of a given population

    Returns:
        (int): The two randomly selected candidates from its population based on probability
    """
    total_fitness = sum(fitness_scores)    
    selection_probability = [fitness / total_fitness for fitness in fitness_scores] 
    return random.choices(population, weights=selection_probability, k=2)

def rank_selection(population, fitness_scores):
    """Ranks all the candidates from the population based on its fitness

    Args:
        population (array): The number of candidates that will be converged in a generation
        fitness_scores (array): The number of candidates that will be converged in a generation

    Returns:
        (int): The two randomly selected candidates from its population based on its fitness value
    """
    sorted_population = [individual for _, individual in sorted(zip(fitness_scores, population), reverse=True)]
    rank_probs = [i / len(sorted_population) for i in range(1, len(sorted_population) + 1)]
    return random.choices(sorted_population, weights=rank_probs, k=2)

def stochastic_universal_sampling(population, fitness_scores):
    """Randomly selects candidates using evenly spaced intervals for weak candidates to be selected

    Args:
        population (array): The two randomly selected candidates from its population based on probability
        fitness_scores (array): The number of candidates that will be converged in a generation

    Returns:
        (array): The selected parents for upcoming generating offspring
    """
    total_fitness = sum(fitness_scores)
    selection_probs = [fitness / total_fitness for fitness in fitness_scores]
    accumulated_probs = [sum(selection_probs[:i+1]) for i in range(len(selection_probs))]
    
    start = random.uniform(0, 1 / len(population))
    pointers = [start + i / len(population) for i in range(len(population))]
    
    selected_parents = []
    for pointer in pointers:
        for i in range(len(accumulated_probs)):
            if pointer <= accumulated_probs[i]:
                selected_parents.append(population[i])
                break
    
    return selected_parents

def tournament_selection(population, fitness_scores):
    """Selects the fittest candidates from the current generation through K-way tournament

    Args:
        population (array): The two randomly selected candidates from its population based on probability
        fitness_scores (array): The number of candidates that will be converged in a generation

    Returns:
       (array): The selected parents for upcoming generating offspring
    """
    TOURNAMENT_SIZE = 5
    selected_parents = []
    for _ in range(2):
        tournament_candidates = random.sample(range(len(population)), TOURNAMENT_SIZE)
        tournament_scores = [fitness_scores[i] for i in tournament_candidates]
        winner_index = tournament_candidates[tournament_scores.index(max(tournament_scores))]
        selected_parents.append(population[winner_index])
    return selected_parents

def elitist_selection(population, fitness_scores):
    """Selects limited number of candidates with the highest fitness values

    Args:
        population (array): The two randomly selected candidates from its population based on probability
        fitness_scores (array): The number of candidates that will be converged in a generation

    Returns:
        elite_individuals(array): The selected candidates that have the highest fitness values
        non_elite_population(array): The rest of the population that are not picked
    """
    ELITE_SIZE = 2
    elite_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)[:ELITE_SIZE]
    elite_individuals = [population[i] for i in elite_indices]
    non_elite_population = [individual for i, individual in enumerate(population) if i not in elite_indices]
    return elite_individuals, non_elite_population

def crossover(parent1, parent2):
    """Combines the genetic information of two parents to generate new offspring

    Args:
        parent1 (array): The first parent of the offspring
        parent2 (array): The second parent of the offspring

    Returns:
        offspring1 (array): The first offspring after combination
        offspring2 (array): The second offspring after combination
    """
    point = random.randint(4, len(parent1) - 1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]
    return offspring1, offspring2

def insertion_mutation(individual, mutation_rate):
    """Randomly selects a location of the gene and replaces the value

    Args:
        individual (array): A candidate from a population in its generation
        mutation_rate (float): The probability of a likelihood of a mutation

    Returns:
        (array): A modified candidate from the genetic operator
    """
    mutated = individual[:]
    for i in range(4, len(mutated) - 1):
        if random.random() < mutation_rate:
            mutated[i] = random.randint(0, 7)
    return mutated

def inversion_mutation(individual, mutation_rate):
    """Randomly choses two points of the gene and inverts the values

    Args:
        individual (array): A candidate from a population in its generation
        mutation_rate (float): The probability of a likelihood of a mutation

    Returns:
        (array): A modified candidate from the genetic operator 
    """
    mutated = individual[:]
    position1, position2 = sorted(random.sample(range(4, len(individual) - 1), 2))
    if random.random() < mutation_rate:
        mutated[position1:position2 + 1] = reversed(individual[position1:position2 + 1])
    return mutated

def scramble_mutation(individual, mutation_rate):
    """Randomly selects two points of the gene and places the values randomly in the following two points

    Args:
        individual (array): A candidate from a population in its generation
        mutation_rate (float): The probability of a likelihood of a mutation

    Returns:
       (array): A modified candidate from the genetic operator 
    """
    mutated = individual[:]
    position1, position2 = sorted(random.sample(range(4, len(individual) - 1), 2))
    subset = mutated[position1:position2 + 1]
    random.shuffle(subset)
    if random.random() < mutation_rate:
        mutated[position1:position2 + 1] = subset
    return mutated

def swap_mutation(individual, mutation_rate):
    """Randomly selects two genes and swap its location for both genes

    Args:
        individual (array): A candidate from a population in its generation
        mutation_rate (float): The probability of a likelihood of a mutation

    Returns:
        (array): A modified candidate from the genetic operator 
    """
    mutated = individual[:]
    point1 = random.randint(4, len(individual) - 1)
    point2 = random.randint(4, len(individual) - 1)
    point1_value = mutated[point1]
    point2_value = mutated[point2]
    if random.random() < mutation_rate:
        mutated[point1]  = individual[point2]
        mutated[point2]  = individual[point1]
    return mutated 

def random_mutation_type(individual, mutation_rate):
    """Randomly selects any type of mutation genetic operator for generating offspring

    Args:
        individual (array): A candidate from a population in its generation
        mutation_rate (float): The probability of a likelihood of a mutation

    Returns:
        mutation_method(individual, mutation_rate): The selected method for performing mutation genetic operator 
    """
    random_type = random.randint(0, 3)
    match random_type: 
        case 0:
            return insertion_mutation(individual, mutation_rate)
        case 1:
            return inversion_mutation(individual, mutation_rate)
        case 2:
            return scramble_mutation(individual, mutation_rate)
        case 3:
            return swap_mutation(individual, mutation_rate)

def mutate(mutation_type, individual, mutation_rate):
    """Selects a mutation type for generating offspring based of user's preference

    Args:
        mutation_type (str): 
        individual (array): A candidate from a population in its generation
        mutation_rate (float): The probability of a likelihood of a mutation

    Returns:
        mutation_method(individual, mutation_rate): The selected method for performing mutation genetic operator 
    """
    match mutation_type:
        case "Insertion Mutation":
            return insertion_mutation(individual, mutation_rate)
        case "Inversion Mutation":
            return inversion_mutation(individual, mutation_rate)
        case "Scramble Mutation":
            return scramble_mutation(individual, mutation_rate)
        case "Swap Mutation":
            return swap_mutation(individual, mutation_rate)
        case "All of the Above":
            return random_mutation_type(individual, mutation_rate)

def calculate_best_individual(fitness_scores):
    """Calculates the candidate with the highest fitness value

    Args:
        fitness_scores (array): The fitness values of each candidate

    Returns:
        max_fitness (int): The highest fitness value of its population
        best_individual (array): The candidate with the highest fitness value of the population
    """
    max_fitness = max(fitness_scores)
    best_individual = population[fitness_scores.index(max_fitness)]
    print(f"Best Individual: Candidate {fitness_scores.index(max_fitness)}") 
    print(f"Chromosome of Best Individual: {best_individual}")
    print(f"Fitness of Best Individual: {max_fitness}")
    return max_fitness, best_individual 

def best_individual_as_array(best_individual_per_generation, best_individual, max_fitness, fitness_scores, generation):
    """Appends the candidate with the highest fitness value from the top five selection to an array per generation

    Args:
        best_individual_per_generation (array): The selected candidate with the highest fitness value of its generation
        best_individual (array): The candidate with the highest fitness value of the population
        max_fitness (int): The highest fitness value of its population
        fitness_scores (array): The fitness values of each candidate
        generation (int): Iteration of generating the next population
    """
    best_individual_array = [[fitness_scores.index(max_fitness)], [best_individual], [max_fitness], [generation]]
    best_individual_per_generation.append(best_individual_array)

def sort_top_five_individuals(fitness_scores):
    """Sorts the top five candidates based on its fitness scores

    Args:
        fitness_scores (array): The fitness values of each candidate

    Returns:
        top_five_individual (array): The candidates that are on the top five of highest fitness values
        top_five_individual_indexes (array): The following indexs of candidates from the top five individual array
    """
    sorted_fitness_scores = sorted(fitness_scores, reverse=False)
    top_five_individuals = list(set(sorted_fitness_scores[:5]))
    top_five_individuals_indexes = []
    return top_five_individuals, top_five_individuals_indexes

def selecting_top_five_individual(individual, order, population, fitness_scores):
    """Selects the candidates that have the highest fitness values on the top five

    Args:
        individual (array): A candidate from a population in its generation
        order (int): The order of the candidate selected from top five individual array
        population (array): The two randomly selected candidates from its population based on probability
        fitness_scores (array): The fitness values of each candidate

    Returns:
        selected_individual (array): The individual that is selected for the upcoming top five individuals
        individual_index (int): The index of a selected individual from its population
    """
    selected_individual = population[fitness_scores.index(individual)] 
    individual_index = find_index(population, selected_individual)
    print(f"Candidate {order}: {individual_index}")
    print(f"Chromosome: {selected_individual}")
    return selected_individual, individual_index

def create_MIDI(export_directory, order, i, selected_individual, octave):
    """Produces a MIDI track based on the user's preference

    Args:
        export_directory (str): The file path of the directory for exporting MIDI tracks
        order (int): The order of which the top five individuals are selected for MIDI generation
        i (int): The generation of which the individual is selected
        selected_individual (array): The individual that is selected for the upcoming top five individuals 
        octave (int): The distance between one note and the next note
    """
    music = Music()
    music.generate_MIDI(export_directory, order, i, selected_individual, octave)

def adding_individual_to_top_five(top_five_individuals_indexes, individual_index, population, order):
    """Appends the selected individual to the top five if the individual has the highest fitness value

    Args:
        top_five_individuals_indexes (array): The following indexs of candidates from the top five individual array
        individual_index (int): The index of a selected individual from its population
        population (array): The two randomly selected candidates from its population based on probability
        order (int): The order of the candidate selected from top five individual array 

    Returns:
        (int): The order of the candidate selected from top five individual array 
    """
    top_five_individuals_indexes.append(individual_index)
    population[individual_index] = 0
    order += 1
    return order
           
def find_index(arr, value):
    """Finds the index of a given value

    Args:
        arr (array): The array specified through its argument
        value (int): The specified value from its array

    Returns:
        (int): The value of the index
    """
    try:
        index = arr.index(value)
        return index
    except ValueError:
        return None

def replacing_value_of_population_index(population, new_five_individuals):
    """Replaces a value from a specified index if the following candidate equates to 0

    Args:
        population (array): The order of the candidate selected from top five individual array 
        new_five_individuals (array): The new generated individuals for upcoming replacement for population

    Returns:
        array: Return the new population modified after replacement
    """
    j = 0
    p = 0
    while p < len(population):
        if population[p] == 0:
            population[p] = new_five_individuals[j]
            j += 1
        p += 1
        
    return population

def print_sorted_highest_individual(best_individual_per_generation):
    """Print the candidate that has the highest fitness value of all generation

    Args:
        best_individual_per_generation (array): The selected candidate with the highest fitness value of its generation
    """
    sorted_highest_individual = sorted(best_individual_per_generation, key=lambda x: x[2], reverse=True)
    print(f"Candidate Index: {sorted_highest_individual[-1][0]}")
    print(f"Chromosome: {sorted_highest_individual[-1][1]}")
    print(f"Fitness Score: {sorted_highest_individual[-1][2]}")
    print(f"Generation: {sorted_highest_individual[-1][3]}")

def evolution(export_directory, selection_method: str, generation: int, population: int, mutation_rate: float, mutation_type: str, octave: int):
    """Executes the evolution stage to generate new offsprings and measure the fitness value of each candidate per generation

    Args:
        export_directory (str): The file path of the directory for exporting MIDI tracks
        selection_method (str): The method of how candidates are chosen from a given population
        generation (int): Iteration of generating the next population
        population (int): The order of the candidate selected from top five individual array
        mutation_rate (float): The probability of a likelihood of a mutation
        mutation_type (str): The type of mutation genetic operator to generate offspring
        octave (int): The distance between one note and the next note
    """
    best_individual_per_generation = [] 
    
    print("Selection Method: ", selection_method)
    for GENERATION in range(generation):
        i = GENERATION + 1
        print("\n")
        print("=================================================================================================================")
        print(f"--- GENERATION {i} ---") 
        fitness_scores = [calculate_fitness(individual) for individual in population]
        max_fitness, best_individual = calculate_best_individual(fitness_scores)
        best_individual_as_array(best_individual_per_generation, best_individual, max_fitness, fitness_scores, i)
        top_five_individuals, top_five_individuals_indexes = sort_top_five_individuals(fitness_scores)
       
        print("---------------------------------------------------------------") 
        print(f"Selecting top five candidates of Generation {i}")
        print("---------------------------------------------------------------") 
        order = 1
        for individual in top_five_individuals:
            selected_individual, individual_index = selecting_top_five_individual(individual, order, population, fitness_scores)
            create_MIDI(export_directory, order, i, selected_individual, octave)
            order = adding_individual_to_top_five(top_five_individuals_indexes, individual_index, population, order)
        print("Finishing selecting the top five candidates of the population..")
        print("---------------------------------------------------------------") 
       
        new_five_individuals = [generate_individual(note_quantity, mode, key, tempo) for _ in range(5)]
        population = replacing_value_of_population_index(population, new_five_individuals)
        
        num_to_replace = int(population_size * replacement_rate)
        replace_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:num_to_replace] 
        offspring = [] 
        for i in range(num_to_replace):
            parents = selection(selection_method, num_to_replace, population, fitness_scores) 
            print("Selected parents:") 
            for parent in parents:
                print(parent)
            offspring1, offspring2 = crossover(parents[0], parents[1])
            offspring1 = mutate(mutation_type, offspring1, mutation_rate)
            offspring2 = mutate(mutation_type, offspring2, mutation_rate)
            offspring.append(offspring1)
            offspring.append(offspring2)
        
        for index, offspring_index in enumerate(replace_indices):
            population[offspring_index] = offspring[index]  
        print("=================================================================================================================")
    print("\n")    
    print("FITTEST INDIVIDUAL OF ALL GENERATIONS: ")
    print_sorted_highest_individual(best_individual_per_generation)
    
def generating_population(note_quantity, mode, key, tempo, population_size):
    """Produces a population based on the user's preference

    Args:
        note_quantity (int): The amount of notes that will be performed from a musical piece
        mode (str): The musical scale that are defined by their starting note or tonic
        key (int): Group of pitches or scale that requires the basis of a musical composition
        tempo (int): The speed or pace of a given piece it will be performed in
        population_size (array): The number of candidates that will be converged in a generation

    Returns:
        array: The group of candidates that represents an iteration of a generation
    """
    population = [generate_individual(note_quantity, mode, key, tempo) for _ in range(population_size)]
    print("Generating population...")
    i = 1
    while i <= len(population):
        for individual in population:
            print(f"Individual {i}: {individual}")
            i += 1
    print("Finished generating population...")
    return population

def arguments():
    """Passes the given arguments as values for preparing the generation of candidates

    Returns:
        export_directory (str): The file path of the directory for exporting MIDI tracks
        mode (str): The musical scale that are defined by their starting note or tonic
        key (int): Group of pitches or scale that requires the basis of a musical composition
        tempo (int): The speed or pace of a given piece it will be performed in
        note_quantity (int): The amount of notes that will be performed from a musical piece
        octave (int): The distance between one note and the next note 
        population_size (int): 
        generation (int): Iteration of generating the next population
        selection_type (str): The method of how candidates are chosen from a given population
        mutation_rate (float): The probability of a likelihood of a mutation
        mutation_type (str): The type of mutation genetic operator to generate offspring
    """
    export_directory = os.environ.get('export_directory', None)
    mode = os.environ.get('mode', None)
    key = os.environ.get('key', None)
    tempo_input = os.environ.get('tempo', None)
    note_quantity_input = os.environ.get('note_quantity', None)
    octave_input = os.environ.get('octave', None)
    population_size_input = os.environ.get('population_size', None)
    generation_input = os.environ.get('generation', None)
    selection_type = os.environ.get('selection_type', None)
    mutation_rate_input = os.environ.get('mutation_rate', None)
    mutation_type = os.environ.get('mutation_type', None)

    tempo = int(tempo_input)
    note_quantity = int(note_quantity_input)
    octave = int(octave_input)
    population_size = int(population_size_input)
    generation = int(generation_input)
    mutation_rate = float(mutation_rate_input)
    
    print(f"Export Directory: {export_directory} {type(export_directory)}") 
    print(f"Mode: {mode} {type(mode)}")
    print(f"Key: {key} {type(key)}")
    print(f"Tempo: {tempo} {type(tempo)}")
    print(f"Note Quantity: {note_quantity} {type(note_quantity)}")
    print(f"Octave: {octave} {type(octave)}")
    print(f"Population Size: {population_size} {type(population_size)}")
    print(f"Generation: {generation} {type(generation)}")
    print(f"Selection Type: {selection_type} {type(selection_type)}")
    print(f"Mutation Rate: {mutation_rate} {type(mutation_rate)}")
    print(f"Mutation Type: {mutation_type} {type(mutation_type)}")
    
    return export_directory, mode, key, tempo, note_quantity, octave, population_size, generation, selection_type, mutation_rate, mutation_type

if __name__ == '__main__':
    print("======== Genetic Algorithm ========")
    print("============  RiffGen  ============")
    export_directory, mode, key, tempo, note_quantity, octave, population_size, generation, selection_type, mutation_rate, mutation_type = arguments()
    individual_length = 0
    population = generating_population(note_quantity, mode, key, tempo, population_size)
    evolution(export_directory, selection_type, generation, population, mutation_rate, mutation_type, octave)