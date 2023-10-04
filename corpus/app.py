"""
RiffGen App

This script allows the user to launch a desktop application that provides the user to 
generate melodic music as a MIDI file through each generation. The user is required 
to select the following values from the selection of dropdown menus that shares the scope 
of music. 

The following lists required to producing MIDI music includes:
    * Mode - The musical scale that are defined by their starting note or tonic (Ionian, Aeolian, Lydian and etc).
    * Key - Group of pitches or scale that requires the basis of a musical composition (C, D and etc).
    * Tempo - The speed or pace of a given piece it will be performed in.
    * Note Quantity - The amount of notes that will be performed from a musical piece.
    * Octave - The interval between one musical pitch and another with double its frequency.
   
Once the user has finished selecting their musical scope for the piece to be generated,
the user is also required to select their preference for performing Genetic Algorithm,
with also the following list required:
    * Population Size - The number of candidates that will be converged in a generation.
    * Generation - The iteration of generation for candidates to be converged as a cycle.
    * Selection Type - The method of how candidates are chosen from a given population.
    * Mutation Rate - The probability of the likelihood that an individual will undergo the mutation process.
    * Mutation Type - The type of mutation operator used to maintain genetic diversity.

The following script contains the following classes:
    * MainWindow(QWidget) - The main window that will display the given list of parameters from music and Genetic Algorithm 
    * Dialog(QDialog) - A dialog that will display the console log of the Genetic Algorithm process after execution.
"""

import os
import sys 
from PyQt6.QtWidgets import (
    QApplication, QDialog, QComboBox, QGridLayout, QHBoxLayout, QPushButton, QLabel, QWidget,
    QLineEdit, QMessageBox, QPlainTextEdit, QFileDialog
)   
from PyQt6.QtGui import *
from PyQt6.QtCore import *

from functools import partial # efficient for passing parameters through functions

import argparse
import numpy as np
from music import *

# ---------------------- Changes Taskbar Icon ------------------------
import ctypes
myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
# --------------------------------------------------------------------

export_directory = ""
mode = ""
key = ""
tempo = ""
note_quantity = ""
octave = ""
mutation_rate = ""
mutation_type = ""
population_size = ""
selection_type = ""
generation_iteration = ""
replacement_rate = ""

class MainWindow(QWidget):
    """
    A class used to initialize the main window
    
    ...
    
    Methods
    -------
    current_mode(self, _)
        Prints the selected mode and sets the selected option to the global variable.

    current_key(self, _)
        Prints the selected key and sets the selected option to the global variable.
    
    current_tempo(self, tempo_text)
        Prints the tempo from user input and sets the selected option to the global variable.
    
    curent_note_quantity(self, note_quantity)
        Prints the note quantity from user input and sets the selected option to the global variable.

    current_octave(self, octave)
        Prints the octave from user input and sets the selected option to the global variable.
    
    current_mutation_rate(self, mutation_rate_text)
        Prints the mutation rate from user input and sets the selected option to the global variable.
    
    current_mutation_type(self, _)
        Prints the selected mutation type and sets the selected option to the global variable.
    
    current_population_size(self, population_size_text)
        Prints the population size from user input and sets the selected option to the global variable.

    current_selection_type(self, _)
        Prints the selected selection type and sets the selected option to the global variable.
    
    current_generation_iteration(self, generation_iteration_text)
        Prints the generation iteration from user input and sets the selected option to the global variable.
    
    generate_lick(self, mode, key, tempo, note_quantity, octave, mutation_rate, mutation_type, population_size, selection_type, generation_iteration)
        Initializes the dialog window if the following values meet the requirements for proceeding the genetic algorithm. Returns 0 if 
        the following parameters does not align with the requirements.
    
    """ 
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set App Widget
        self.setWindowTitle("RiffGen")
        
        # print QPlainTextEdit by which cannot be edited, which should be only Read Only
        self.ga_log = QPlainTextEdit(self)
        self.ga_log.setReadOnly(True)
        self.ga_log.setPlaceholderText("Output")
        
        # === MUSIC PARAMETERS ===
        # Set dropdown for Musical Modes
        self.modes_dropdown = QComboBox()
        self.modes_dropdown.setPlaceholderText("Modes")
        self.modes_dropdown.addItems([
            'Major', 'Dorian', 'Phrygian', 'Lydian', 'Mixolydian', 'Minor', 'Locrian',
            'Harmonic Minor', 'Melodic Minor', 'Neapolitan Minor', 'Hungarian Minor', 
            'Pentatonic Major', 'Pentatonic Minor', 'Blues'
        ])
        self.modes_dropdown.activated.connect(self.current_mode)

        # Set Dropdown for Key
        self.key_dropdown = QComboBox()
        self.key_dropdown.setPlaceholderText("Key")
        self.key_dropdown.addItems([
            'C', 'C# / Db', 'D', 'D# / Eb', 'E', 'F', 'F# / Gb', 'G', 'G# / Ab', 'A', 'A# / Bb', 'B', 'C'
        ])
        self.key_dropdown.activated.connect(self.current_key) # key dropdown function
        
        # Text-based input for Tempo
        self.tempo_input = QLineEdit()
        self.tempo_input.setValidator(QIntValidator())
        self.tempo_input.setMaxLength(3)
        self.tempo_input.setPlaceholderText("Tempo")
        self.tempo_input.textChanged.connect(self.current_tempo) # current tempo function

        # Set input for note quantity
        self.note_quantity_input = QLineEdit()
        self.note_quantity_input.setValidator(QIntValidator())
        self.note_quantity_input.setMaxLength(3)
        self.note_quantity_input.setPlaceholderText("Quantity of Notes")
        self.note_quantity_input.textChanged.connect(self.current_note_quantity) # note quantity function

        # Set input for maximum octave range
        self.octave_input = QLineEdit()
        self.octave_input.setValidator(QIntValidator())
        self.octave_input.setMaxLength(1)
        self.octave_input.setPlaceholderText("Octave Level")
        self.octave_input.textChanged.connect(self.current_octave) # octave function

        # === GENETIC ALGORITHM ===
        # Set length of crossover
        self.mutation_rate_input = QLineEdit()
        self.mutation_rate_input.setValidator(QDoubleValidator(0.0, 1.0, 6))
        self.mutation_rate_input.setPlaceholderText("Mutation Rate")
        self.mutation_rate_input.textChanged.connect(self.current_mutation_rate)

        # Set mutation type from dropdown
        self.mutation_type_dropdown = QComboBox()
        self.mutation_type_dropdown.setPlaceholderText("Mutation Type")
        self.mutation_type_dropdown.addItems([
            'Insertion Mutation', 'Inversion Mutation', 'Scramble Mutation', 'Swap Mutation', 
            'All of the Above'
        ])
        self.mutation_type_dropdown.activated.connect(self.current_mutation_type) # mutation type function

        # Set population size
        self.population_size_input = QLineEdit()
        self.population_size_input.setValidator(QIntValidator())
        self.population_size_input.setMaxLength(3)
        self.population_size_input.setPlaceholderText("Population Size")
        self.population_size_input.textChanged.connect(self.current_population_size) # population size function

        # Set type of termination criteria 
        self.selection_type_dropdown = QComboBox()
        self.selection_type_dropdown.setPlaceholderText("Selection Type")
        self.selection_type_dropdown.addItems([
            'Roulette Wheel Selection', 'Rank Selection', 'Tournament Selection', 'Elitist Selection',
        ])
        self.selection_type_dropdown.activated.connect(self.current_selection_type) # selection type function

        # Set iteration for generation
        self.generation_iteration_input = QLineEdit()
        self.generation_iteration_input.setValidator(QIntValidator())
        self.generation_iteration_input.setMaxLength(3)
        self.generation_iteration_input.setPlaceholderText("Generation Iterations")
        self.generation_iteration_input.textChanged.connect(self.current_generation_iteration)

        # Set replacement rate
        self.replacement_rate_input = QLineEdit()
        self.replacement_rate_input.setValidator(QDoubleValidator(0.0, 1.0, 6))
        self.replacement_rate_input.setPlaceholderText("Replacement Rate")
        self.replacement_rate_input.textChanged.connect(self.current_replacement_rate)
        
        # Adding Export Directory for exporting MIDI tracks at a specified location
        self.export_directory_btn = QPushButton()
        self.export_directory_btn.setText("Export Directory")
        self.export_directory_btn.clicked.connect(lambda: self.exportDirectoryDialog())

        # Set Generate Lick Button
        self.generate_button = QPushButton()
        self.generate_button.setText("Run")
        self.generate_button.clicked.connect(lambda: self.generate_lick(
            mode, key, tempo, note_quantity, octave, mutation_rate, mutation_type, population_size, selection_type, generation_iteration, replacement_rate
        ))
        self.generate_button.setEnabled(False)
        
        # GA process 
        self.ga = GA()
        self.ga.output_ready.connect(self.on_outputReady)

        # Set Grid Layout for all widgets
        layout = QGridLayout()
        # Music Widgets
        layout.addWidget(self.ga_log, 0, 0, 1, 5)
        layout.addWidget(self.modes_dropdown, 1, 0)
        layout.addWidget(self.key_dropdown, 1, 1)
        layout.addWidget(self.tempo_input, 1, 2)
        layout.addWidget(self.note_quantity_input, 1, 3)
        layout.addWidget(self.octave_input, 1, 4)
        # Genetic Algorithm Widgets
        layout.addWidget(self.mutation_rate_input, 2, 0)
        layout.addWidget(self.mutation_type_dropdown, 2, 1)
        layout.addWidget(self.population_size_input, 2, 2)
        layout.addWidget(self.selection_type_dropdown, 2, 3)
        layout.addWidget(self.generation_iteration_input, 2, 4)
        layout.addWidget(
            self.replacement_rate_input, 3, 0, 1, 5
        )
        layout.addWidget(
            self.export_directory_btn, 4, 0, 1, 5
        )
        layout.addWidget(
            self.generate_button, 5, 0, 1, 5
        )

        # Set layout for all widgets
        self.setLayout(layout)
      
    def current_mode(self, _):
        """Prints the selected mode and sets the selected option to the global variable.
       
        Parameters
        ----------
        _ : str
            The string of a music mode that is selected     
        """
        global mode
        mode_text = self.modes_dropdown.currentText()
        print("Mode: ", mode_text)
        mode = mode_text
        print(mode)
        
    def current_key(self, _):
        """Prints the selected key and sets the selected option to the global variable.
        
        Parameters
        ----------
        _ : str
            The string of a key that is selected 
        """  
        global key
        key_text = self.key_dropdown.currentText()
        print("Key: ", key_text)
        key = key_text
        print(key)
        
    def current_tempo(self, tempo_text):
        """Prints the tempo from user input and sets the selected option to the global variable.

        Parameters
        ----------
        tempo_text : str
            The string of a tempo that is inputed
        """
        global tempo
        print("Tempo: ", tempo_text) 
        tempo = tempo_text
        print(tempo)
       
    def current_note_quantity(self, note_text):
        """Prints the note quantity from user input and sets the selected option to the global variable.

        Args:
            note_text (str): The string of the specifed note quantity from user input
        """
        global note_quantity
        print("Note Quantity: ", note_text)
        note_quantity = note_text
        print(note_quantity)
        
    def current_octave(self, octave_text):
        """Prints the octave from user input and sets the selected option to the global variable.

        Args:
            octave_text (str): The string of the specified octave from user input
        """
        global octave
        print("Octave: ", octave_text)
        octave = octave_text
        print(octave)
        
    def current_mutation_rate(self, mutation_rate_text):
        """Prints the mutation rate from user input and sets the selected option to the global variable.

        Args:
            mutation_rate_text (str): The string of the specified mutation rate from user input
        """
        global mutation_rate
        print("Mutation Rate: ", mutation_rate_text) 
        mutation_rate = mutation_rate_text
        print(mutation_rate)
        
    def current_mutation_type(self, _):
        """Prints the selected mutation type and sets the selected option to the global variable.

        Args:
            _ (str): The string of the selected mutation type from dropdown menu
        """
        global mutation_type
        mutation_type_text = self.mutation_type_dropdown.currentText()
        print("Mutation Type: ", mutation_type_text)
        mutation_type = mutation_type_text
        print(mutation_type)
        
    def current_population_size(self, population_size_text):
        """Prints the population size from user input and sets the selected option to the global variable.

        Args:
            population_size_text (str): The string of specified population size from user input
        """
        global population_size
        print("Population Size: ", population_size_text) 
        population_size = population_size_text
        print(population_size)
    
    def current_selection_type(self, _):
        """Prints the selected selection type and sets the selected option to the global variable.

        Args:
            _ (str): The string of a selected selection type from dropdown menu
        """
        global selection_type
        selection_type_text = self.selection_type_dropdown.currentText()
        print("Selection Type: ", selection_type_text)
        selection_type = selection_type_text
        print(selection_type)
        
    def current_generation_iteration(self, generation_iteration_text):
        """Prints the generation iteration from user input and sets the selected option to the global variable.

        Args:
            generation_iteration_text (str): The string of a specified iteration of generation from user input
        """
        global generation_iteration
        print("Generation Iteration: ", generation_iteration_text)
        generation_iteration = generation_iteration_text
        print(generation_iteration)
        
    def current_replacement_rate(self, replacement_rate_text):
        """Prints the generation iteration from user input and sets the selected option to the global variable.

        Args:
            generation_iteration_text (str): The string of a specified iteration of generation from user input
        """
        global replacement_rate 
        print("Replacement Rate: ", replacement_rate_text)
        replacement_rate = replacement_rate_text
        print(replacement_rate)
    # ---------------------------------------------
    
    def on_readyReadStandardOutput(self):
        """Reads output using QProcess
        """
        # Called when the QProcess has new output to read
        output = self.process.readAllStandardOutput().data().decode()
        self.ga_log.insertPlainText(output)        

    def on_readyReadStandardError(self):
        """Reads error output using QProcess
        """
        # Called when the QProcess has new error output to read
        error_output = self.process.readAllStandardError().data().decode()
        self.ga_log.insertPlainText(error_output)
 
    def on_processFinished(self, exit_code, exit_status):
        """Notifies the user when the QProcess has finished running
        """
        # Called when the QProcess has finished running
        self.run_button.setEnabled(False) 
        msg = QMessageBox() 
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText("The following process is completed")
        msg.setWindowTitle("Completed")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        button = msg.exec()
        print("The process is finished...")
       
    def clear_output(self):
        """Clears the output display from QPlainTextEdit
        """
        # Clear the output displayed in the QPlainTextEdit
        self.ga_log.clear()
        
    @pyqtSlot(str)
    def on_outputReady(self, output):
        self.ga_log.appendPlainText(output) 
    # -----------------------------
    
    def exportDirectoryDialog(self):
        """Directs the user to specify the location of directory for MIDI generation
        """
        global export_directory
        selected_directory = QFileDialog.getExistingDirectory(self, "Select Preferred Location", "")
        if selected_directory:
            export_directory = selected_directory 
            self.generate_button.setEnabled(True)
            print("Selected directory:", selected_directory)
    
    # --- generate lick clicked ---
    def generate_lick(
        self, mode, key, tempo, note_quantity, octave, mutation_rate, mutation_type, population_size, selection_type, generation_iteration, replacement_rate
        ):
        """Initializes the dialog window if the following values meet the requirements for proceeding the genetic algorithm. Returns 0 if 
        the following parameters does not align with the requirements.

        Args:
            mode (str): The musical scale that are defined by their starting note or tonic (Ionian, Aeolian, Lydian and etc)
            key (str): Group of pitches or scale that requires the basis of a musical composition (C, D and etc)
            tempo (str): The speed or pace of a given piece it will be performed in
            note_quantity (str): The amount of notes that will be performed from a musical piece
            octave (str): The interval between one musical pitch and another with double its frequency
            mutation_rate (str): The probability of the likelihood that an individual will undergo the mutation process
            mutation_type (str): The method of how candidates are chosen from a given population
            population_size (str): The number of candidates that will be converged in a generation
            selection_type (str): The method of how candidates are chosen from a given population
            generation_iteration (str): The iteration of generation for candidates to be converged as a cycle

        Returns:
            0: Returns 0 if the following value such as mutation rate is lower than 0 or higher than 1, therefore has to be a float value
        """
        
        print("Generate Lick button pressed")
       
        # check any empty strings 
        parameters = [mode, key, tempo, note_quantity, octave, mutation_rate, mutation_type, population_size, selection_type, generation_iteration, replacement_rate]
        is_empty = [s == '' for s in parameters]
        
        if any(is_empty) == True:
            parameters_name = [
                "Mode", "Key", "Tempo", "Note Quantity", "Octave", "Mutation Rate", "Mutation Type", "Population Size", "Selection Type", "Generation Iteration"
            ]
            empty_string = ""
            
            i = 0  
            while i < len(parameters):
                if is_empty[i] == True:
                    empty_string += parameters_name[i]
                    if i != len(parameters) - 1:
                        empty_string += ", "
                i += 1           
                    
            msg = QMessageBox() 
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setText("Empty values must be inserted to proceed")
            msg.setInformativeText("The following fields that is listed below must be implemented in their fields")
            msg.setDetailedText(f"The following values that are empty are the following: {empty_string}")
            msg.setWindowTitle("Invalid Empty Values")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            button = msg.exec()
            return 0
           
        if float(mutation_rate) > 1 or float(mutation_rate) < 0:
            msg = QMessageBox() 
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setText("The mutation rate must not be lower than 0 or higher than 1")
            msg.setWindowTitle("Mutation Rate - Out of Range")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            button = msg.exec()
            return 0
        
        if float(replacement_rate) > 1 or float(replacement_rate) < 0:
            msg = QMessageBox() 
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setText("The replacement rate must not be lower than 0 or higher than 1")
            msg.setWindowTitle("Replacement Rate - Out of Range")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            button = msg.exec()
            return 0
        
        # ga = GA()
        self.ga_log.clear() 
        self.ga_log.appendPlainText("Running Genetic Algorithm...")
        self.ga.main_function(export_directory, mode, key, tempo, note_quantity, octave, population_size, generation_iteration, selection_type, mutation_rate, mutation_type, replacement_rate)
       
        msg = QMessageBox() 
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText("The process of Genetic Algorithm is complete")
        msg.setWindowTitle("GA Complete")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        button = msg.exec()
        
class GA(QObject):
    """
    Genetic Algorithm

    This class borrows the techniques from natural computation, using Genetic Algorithm to 
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
    
    output_ready = pyqtSignal(str)
    
    def __init__(self):
        # new line
        super().__init__()
        # pass
    
    def insert_mode(self, mode):
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
    
    def insert_key(self, key):
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

    def generate_individual(self, note_quantity, mode, key, tempo):
        """Produces a candidate for the population with user's preferences

        Args:
            note_quantity (int): The amount of notes that will be performed from a musical piece
            mode (str): The musical scale that are defined by their starting note or tonic
            key (int): Group of pitches or scale that requires the basis of a musical composition
            tempo (int): The speed or pace of a given piece it will be performed in

        Returns:
            individual (array): A candidate from a population in its generation
        """
        # individual = [random.randint(0, 7) for _ in range(note_quantity)]
        individual = [random.randint(0, 7) for _ in range(int(note_quantity))]
        individual[0] = self.insert_mode(mode)
        individual[1] = self.insert_key(key)
        individual[2] = int(tempo)
        return individual

    def selection(self, selection_method, num_to_replace, population, fitness_scores):
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
                return self.roulette_wheel_selection(population, fitness_scores)
            case 'Rank Selection':
                return self.rank_selection(population, fitness_scores) 
            case 'Tournament Selection':
                return self.tournament_selection(population, fitness_scores)
            case 'Stochastic Universal Sampling':
                return self.stochastic_universal_sampling(population, fitness_scores) 
            case 'Elitist Selection':
                return self.elitist_selection(population, fitness_scores) 

    def calculate_fitness(self, individual):
        """Calculates the fitness function of an individual

        Args:
            individual (_type_): A candidate from a population in its generation

        Returns:
            sum(individual): The sum of each numerical values from its candidate
        """
        # return sum(individual)
        return sum(individual)

    def roulette_wheel_selection(self, population, fitness_scores):
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

    def rank_selection(self, population, fitness_scores):
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

    def stochastic_universal_sampling(self, population, fitness_scores):
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

    def tournament_selection(self, population, fitness_scores):
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

    def elitist_selection(self, population, fitness_scores):
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

    def crossover(self, parent1, parent2):
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

    def insertion_mutation(self, individual, mutation_rate):
        """Randomly selects a location of the gene and replaces the value

        Args:
            individual (array): A candidate from a population in its generation
            mutation_rate (float): The probability of a likelihood of a mutation

        Returns:
            (array): A modified candidate from the genetic operator
        """
        mutated = individual[:]
        for i in range(4, len(mutated) - 1):
            if random.random() < float(mutation_rate):
                mutated[i] = random.randint(0, 7)
        return mutated

    def inversion_mutation(self, individual, mutation_rate):
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

    def scramble_mutation(self, individual, mutation_rate):
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

    def swap_mutation(self, individual, mutation_rate):
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

    def random_mutation_type(self, individual, mutation_rate):
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
                return self.insertion_mutation(individual, mutation_rate)
            case 1:
                return self.inversion_mutation(individual, mutation_rate)
            case 2:
                return self.scramble_mutation(individual, mutation_rate)
            case 3:
                return self.swap_mutation(individual, mutation_rate)

    def mutate(self, mutation_type, individual, mutation_rate):
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
                return self.insertion_mutation(individual, mutation_rate)
            case "Inversion Mutation":
                return self.inversion_mutation(individual, mutation_rate)
            case "Scramble Mutation":
                return self.scramble_mutation(individual, mutation_rate)
            case "Swap Mutation":
                return self.swap_mutation(individual, mutation_rate)
            case "All of the Above":
                return self.random_mutation_type(individual, mutation_rate)

    def calculate_best_individual(self, population, fitness_scores):
        """Calculates the candidate with the highest fitness value

        Args:
            fitness_scores (array): The fitness values of each candidate

        Returns:
            max_fitness (int): The highest fitness value of its population
            best_individual (array): The candidate with the highest fitness value of the population
        """
        max_fitness = max(fitness_scores)
        best_individual = population[fitness_scores.index(max_fitness)]
        self.output_ready.emit(f"Best Individual: Candidate {fitness_scores.index(max_fitness)}") 
        self.output_ready.emit(f"Chromosome of Best Individual: {best_individual}")
        self.output_ready.emit(f"Fitness of Best Individual: {max_fitness}")
        return max_fitness, best_individual 

    def best_individual_as_array(self, best_individual_per_generation, best_individual, max_fitness, fitness_scores, generation):
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

    def sort_top_five_individuals(self, fitness_scores):
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

    def selecting_top_five_individual(self, individual, order, population, fitness_scores):
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
        individual_index = self.find_index(population, selected_individual)
        self.output_ready.emit(f"Candidate {order}: {individual_index}")
        self.output_ready.emit(f"Chromosome: {selected_individual}")
        return selected_individual, individual_index

    def create_MIDI(self, export_directory, order, i, selected_individual, octave):
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

    def adding_individual_to_top_five(self, top_five_individuals_indexes, individual_index, population, order):
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
            
    def find_index(self, arr, value):
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

    def replacing_value_of_population_index(self, population, new_five_individuals):
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

    def print_sorted_highest_individual(self, best_individual_per_generation):
        """Print the candidate that has the highest fitness value of all generation

        Args:
            best_individual_per_generation (array): The selected candidate with the highest fitness value of its generation
        """
        sorted_highest_individual = sorted(best_individual_per_generation, key=lambda x: x[2], reverse=True)
        self.output_ready.emit(f"Candidate Index: {sorted_highest_individual[-1][0]}")
        self.output_ready.emit(f"Chromosome: {sorted_highest_individual[-1][1]}")
        self.output_ready.emit(f"Fitness Score: {sorted_highest_individual[-1][2]}")
        self.output_ready.emit(f"Generation: {sorted_highest_individual[-1][3]}")

    def evolution(self, export_directory, selection_method: str, generation: int, population, population_size: int, mutation_rate: float, mutation_type: str, octave: int, replacement_rate: float):
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
        
        self.output_ready.emit(f"Selection Method: {selection_method}")
        for GENERATION in range(generation):
            i = GENERATION + 1
            self.output_ready.emit("\n")
            self.output_ready.emit("=================================================================================================================")
            self.output_ready.emit(f"--- GENERATION {i} ---") 
            fitness_scores = [self.calculate_fitness(individual) for individual in population]
            max_fitness, best_individual = self.calculate_best_individual(population, fitness_scores)
            self.best_individual_as_array(best_individual_per_generation, best_individual, max_fitness, fitness_scores, i)
            top_five_individuals, top_five_individuals_indexes = self.sort_top_five_individuals(fitness_scores)
        
            self.output_ready.emit("---------------------------------------------------------------") 
            self.output_ready.emit(f"Selecting top five candidates of Generation {i}")
            self.output_ready.emit("---------------------------------------------------------------") 
            order = 1
            for individual in top_five_individuals:
                selected_individual, individual_index = self.selecting_top_five_individual(individual, order, population, fitness_scores)
                self.create_MIDI(export_directory, order, i, selected_individual, octave)
                order = self.adding_individual_to_top_five(top_five_individuals_indexes, individual_index, population, order)
            self.output_ready.emit("Finishing selecting the top five candidates of the population..")
            self.output_ready.emit("---------------------------------------------------------------") 
        
            new_five_individuals = [self.generate_individual(note_quantity, mode, key, tempo) for _ in range(5)]
            population = self.replacing_value_of_population_index(population, new_five_individuals)
            
            num_to_replace = int(population_size * replacement_rate)
            replace_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:num_to_replace] 
            offspring = [] 
            for i in range(num_to_replace):
                parents = self.selection(selection_method, num_to_replace, population, fitness_scores) 
                self.output_ready.emit("Selected parents:") 
                self.output_ready.emit(f"Parent 1: {parents[0]}")
                self.output_ready.emit(f"Parent 2: {parents[0]}")
                
                # for parent in parents:
                    # self.output_ready.emit(parent)
                offspring1, offspring2 = self.crossover(parents[0], parents[1])
                offspring1 = self.mutate(mutation_type, offspring1, mutation_rate)
                offspring2 = self.mutate(mutation_type, offspring2, mutation_rate)
                offspring.append(offspring1)
                offspring.append(offspring2)
            
            for index, offspring_index in enumerate(replace_indices):
                population[offspring_index] = offspring[index]  
            self.output_ready.emit("=================================================================================================================")
        self.output_ready.emit("\n")    
        self.output_ready.emit("FITTEST INDIVIDUAL OF ALL GENERATIONS: ")
        self.print_sorted_highest_individual(best_individual_per_generation)
        
    def generating_population(self, note_quantity, mode, key, tempo, population_size):
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
        population = [self.generate_individual(note_quantity, mode, key, tempo) for _ in range(population_size)]
        self.output_ready.emit("Generating population...")
        i = 1
        while i <= len(population):
            for individual in population:
                self.output_ready.emit(f"Individual {i}: {individual}")
                i += 1
        self.output_ready.emit("Finished generating population...")
        return population

    def arguments(self, export_directory_input, mode_input, key_input, tempo_input, note_quantity_input, octave_input, population_size_input, generation_input, selection_type_input, mutation_rate_input, mutation_type_input, replacement_rate_input):
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
        
        tempo = int(tempo_input)
        note_quantity = int(note_quantity_input)
        octave = int(octave_input)
        population_size = int(population_size_input)
        generation = int(generation_input)
        mutation_rate = float(mutation_rate_input)
        replacement_rate = float(replacement_rate_input)
        
        self.output_ready.emit(f"Export Directory: {export_directory_input} {type(export_directory_input)}") 
        self.output_ready.emit(f"Mode: {mode_input} {type(mode_input)}")
        self.output_ready.emit(f"Key: {key_input} {type(key_input)}")
        self.output_ready.emit(f"Tempo: {tempo} {type(tempo)}")
        self.output_ready.emit(f"Note Quantity: {note_quantity} {type(note_quantity)}")
        self.output_ready.emit(f"Octave: {octave} {type(octave)}")
        self.output_ready.emit(f"Population Size: {population_size} {type(population_size)}")
        self.output_ready.emit(f"Generation: {generation} {type(generation)}")
        self.output_ready.emit(f"Selection Type: {selection_type_input} {type(selection_type_input)}")
        self.output_ready.emit(f"Mutation Rate: {mutation_rate_input} {type(mutation_rate_input)}")
        self.output_ready.emit(f"Mutation Type: {mutation_type_input} {type(mutation_type_input)}")
        self.output_ready.emit(f"Replacement Rate: {replacement_rate} {type(replacement_rate)}")
        
        return export_directory_input, mode_input, key_input, tempo, note_quantity, octave, population_size, generation, selection_type_input, mutation_rate_input, mutation_type_input, replacement_rate

    def main_function(self, export_directory_input, mode_input, key_input, tempo_input, note_quantity_input, octave_input, population_size_input, generation_input, selection_type_input, mutation_rate_input, mutation_type_input, replacement_rate_input):
        self.output_ready.emit("======== Genetic Algorithm ========")
        self.output_ready.emit("============  RiffGen  ============")
        export_directory, mode, key, tempo, note_quantity, octave, population_size, generation, selection_type, mutation_rate, mutation_type, replacement_rate = self.arguments(export_directory_input, mode_input, key_input, tempo_input, note_quantity_input, octave_input, population_size_input, generation_input, selection_type_input, mutation_rate_input, mutation_type_input, replacement_rate_input)
        individual_length = 0
        population = self.generating_population(note_quantity, mode, key, tempo, population_size)
        self.evolution(export_directory, selection_type, generation, population, population_size, mutation_rate, mutation_type, octave, replacement_rate) 

# Initializing app
app = QApplication([])
# Set App Icon
app_icon = QIcon('resources/default_icon.png') # Locate icon
app.setWindowIcon(app_icon) # Set window icon
# Initializing Main Window
w = MainWindow()
# Display Window
w.show()
# Quit application terminally
sys.exit(app.exec())