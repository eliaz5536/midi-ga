from midiutil.MidiFile import MIDIFile
import os
import pretty_midi
import random

# Extract notes from a genetic algorithm
note_keys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
scale = [0, 1, 2, 3, 4, 5]

# The scales are in E1 note as default
## ==> 0 represents space (break)
E_major_scale = (0, 40, 42, 44, 45, 47, 49, 51)
E_dorian_scale = (0, 40, 42, 43, 45, 47, 49, 50)
E_phrygian_scale = (0, 40, 41, 43, 45, 47, 48, 50)
E_lydian_scale = (0, 40, 42, 44, 46, 47, 49, 51)
E_mixolydian_scale = (0, 40, 42, 44, 45, 47, 49, 50)
E_minor_scale = (0, 40, 42, 43, 45, 47, 48, 50)
E_locrian_scale = (0, 40, 41, 43, 45, 46, 48, 50)
E_harmonic_minor_scale = (0, 40, 42, 43, 45, 47, 48, 51)
E_melodic_minor_scale = (0, 40, 42, 43, 45, 47, 49, 51)
E_neapolitan_minor_scale = (0, 40, 41, 43, 45, 47, 49, 51)
E_hungarian_minor_scale = (0, 40, 42, 43, 46, 47, 48, 51)
E_pentatonic_major_scale = (0, 40, 42, 44, 47, 49)
E_pentatonic_minor_scale = (0, 40, 43, 45, 47, 50)
E_blues_scale = (0, 40, 43, 45, 46, 47, 50)

octaves = [0, 12]

class Music():
    """
    A class that produces MIDI based on the preferences by the user
    
    Methods:
        read_mode(best_individual): Returns the mode based on the numerical value of first array of the candidate
        read_tempo(best_individual): Returns the value of tempo based on the numerical value of third array of the candidate
        key_name(best_individual): Returns the key based on the numerical value of second array of the candidate
        read_key(best_individual): Return the numerical order of the key based on the second array of the candidate
        retrieve_scale(mode): Returns the specified scale as an array based on the string mode
        shift_scale(scale, key): Returns the shifted scale based on the selected scale and key
        chromosome_to_melody(mode, key, octave, tempo, best_individual): Returns the generated notes based on the parameters provided by the user
        generate_track_name(order, generation, mode, key, tempo): Returns the track name based on the parameters provided by the user
        generate_MIDI(export_directory, order, generation, best_individual, octave): Produces a MIDI file of notes added based on the parameters provided by the user
    """
    def __init__(self):
        pass

    def read_mode(self, best_individual):
        """Returns the string of a given mode based on the numerical value of first array of a candidate

        Args:
            best_individual (array): The fittest candidate from the given generation

        Returns:
            mode: The musical scale that are defined by their starting note or tonic.
        """
        mode = best_individual[0]
        match mode:
            case 0:
                return "Major"
            case 1:
                return "Dorian"
            case 2:
                return "Phrygian"
            case 3:
                return "Lydian"
            case 4:
                return "Mixolydian"
            case 5:
                return "Minor"
            case 6:
                return "Locrian"
            case 7:
                return "Harmonic Minor"
            case 8:
                return "Melodic Minor"
            case 9:
                return "Neapolitan Minor"
            case 10:
                return "Hungarian Minor"
            case 11:
                return "Pentatonic Major"
            case 12:
                return "Pentatonic Minor"
            case 13:
                return "Blues"
    
    def read_tempo(self, best_individual):
        """Returns the value of tempo based on the numerical value of third array of the candidate

        Args:
            best_individual (array): The fittest candidate from the given generation

        Returns:
            tempo (int): The speed or pace of a given piece it will be performed in
        """
        tempo = best_individual[2]
        return tempo
   
    def key_name(self, best_individual):
        """Returns the key based on the numerical value of second array of the candidate

        Args:
            best_individual (array): The fittest candidate from the given generation

        Returns:
            key (str): Group of pitches or scale that requires the basis of a musical composition
        """
        key = best_individual[1]
        match key:
            case 0:
                return "C"
            case 1:
                return "C# / Db"
            case 2:
                return "D"
            case 3:
                return "D# / Eb"
            case 4:
                return "E"
            case 5:
                return "F"
            case 6:
                return "F# / Gb"
            case 7:
                return "G"
            case 8:
                return "G# / Ab"
            case 9:
                return "A"
            case 10:
                return "A# / Bb"
            case 11:
                return "B"
            
    def read_key(self, best_individual):
        """Return the numerical order of the key based on the second array of the candidate

        Args:
            best_individual (array): The fittest candidate from the given generation

        Returns:
            key (int): Group of pitches or scale that requires the basis of a musical composition
        """
        key = best_individual[1]
        match key:
            case 0:
                return 8 # C
            case 1:
                return 9 # C#
            case 2:
                return 10 # D
            case 3:
                return 11 # D#
            case 4:
                return 0 # E
            case 5:
                return 1 # F
            case 6:
                return 2 # F#
            case 7:
                return 3 # G
            case 8:
                return 4 # G#
            case 9:
                return 5 # A
            case 10:
                return 6 # A#
            case 11:
                return 7 # B
                
    def retrieve_scale(self, mode):
        """Returns the specified scale as an array based on the string mode

        Args:
            mode (str): Returns the specified scale as an array based on the string mode

        Returns:
           scale (array): Collection of notes that are played one after another following a set pattern of intervals 
        """
        match mode:
            case "Major":
                return E_major_scale
            case "Dorian":
                return E_dorian_scale
            case "Phrygian":
                return E_phrygian_scale
            case "Lydian":
                return E_lydian_scale
            case "Mixolydian":
                return E_mixolydian_scale
            case "Minor":
                return E_minor_scale
            case "Locrian":
                return E_locrian_scale
            case "Harmonic Minor":
                return E_harmonic_minor_scale
            case "Melodic Minor":
                return E_melodic_minor_scale
            case "Neapolitan Minor":
                return E_neapolitan_minor_scale
            case "Hungarian Minor":
                return E_hungarian_minor_scale
            case "Pentatonic Major":
                return E_pentatonic_major_scale
            case "Pentatonic Minor":
                return E_pentatonic_minor_scale
            case "Blues":
                return E_blues_scale
          
    def shift_scale(self, scale, key):
        """Returns the shifted scale based on the selected scale and key

        Args:
            scale (array): Collection of notes that are played one after another following a set pattern of intervals 
            key (int): Group of pitches or scale that requires the basis of a musical composition

        Returns:
            new_scale (array): Returns the given scale that is shifted
        """
        new_scale = []
        for s in scale:
            if s == 0:
                new_scale.append(0)
                continue
            new_scale.append(s+key)
        return new_scale
        
    def chromosome_to_melody(self, mode, key, octave, tempo, best_individual):
        """Returns the generated notes based on the parameters provided by the user

        Args:
            mode (str): The musical scale that are defined by their starting note or tonic
            key (int): Group of pitches or scale that requires the basis of a musical composition
            octave (str): The interval between one musical pitch and another with double its frequency
            tempo (int): The speed or pace of a given piece it will be performed in 
            best_individual (array): The fittest candidate from the given generation

        Returns:
            chromosome_notes (Array): The notes that were appended based on the selected scale and the range of octave
        """
        scale = self.retrieve_scale(mode)
        key = self.read_key(best_individual)
        shifted_scale = self.shift_scale(scale, key)
        notes = best_individual[3:]
        chromosome_notes = []
        for n in notes:
            octave_range = random.randint(1, octave)
            chromosome_notes.append(shifted_scale[n] * octave_range)
        return chromosome_notes

    def generate_track_name(self, order, generation, mode, key, tempo):
        """Returns the track name based on the parameters provided by the user

        Args:
            order (int): The order of the candidate that is selected from the top five
            generation (int): The iteration of generation that the candidate is selected from
            mode (str): The musical scale that are defined by their starting note or tonic
            key (str): Group of pitches or scale that requires the basis of a musical composition 
            tempo (int): The speed or pace of a given piece it will be performed in 

        Returns:
            track_name (str): Returns the name of the track for MIDI generatino
        """
        track_name = f"{order}_{mode}_{key}_{tempo}_Generation-{generation}"
        return track_name

    def generate_MIDI(self, export_directory, order, generation, best_individual, octave):
        """Produces a MIDI file of notes added based on the parameters provided by the user

        Args:
            export_directory (str): The directory for MIDI files to be exported in
            order (int): The order of the candidate that is selected from the top five 
            generation (str): The order of the candidate that is selected from the top five 
            best_individual (array): The fittest candidate from the given generation
            octave (str): The interval between one musical pitch and another with double its frequency
        """
        # print("Generating music file...")
        mode = self.read_mode(best_individual)
        key = self.read_key(best_individual)
        name_of_key = self.key_name(best_individual)
        tempo = self.read_tempo(best_individual)
        track_name = self.generate_track_name(order, generation, mode, name_of_key, tempo)
        notes = self.chromosome_to_melody(mode, key, octave, tempo, best_individual)
        
        # create your MIDI object
        mf = MIDIFile(1)     # only 1 track
        track = 0   # the only track
        time = 0    # start at the beginning
        mf.addTrackName(track, time, "sample_midi")
        mf.addTempo(track, time, tempo)

        # adding notes
        channel = 0
        volume = 100
        duration = 1 
        
        for i, pitch in enumerate(notes):
            if pitch == 0:
                mf.addNote(track, channel, pitch, time + i, duration, 0)
                continue
            mf.addNote(track, channel, pitch, time + i, duration, volume)

        generation_directory = f"generation_{generation}"
        path = os.path.join(export_directory, generation_directory)
        if os.path.exists(path) == False:
            os.mkdir(path)

        with open(f"{export_directory}\{generation_directory}\{track_name}.mid", 'wb') as outf:
            mf.writeFile(outf)