import mesa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import itertools # use this if needed
import string

WORD_LENGTH = 5

def num_total_words(model):
    total_count = 0
    for agent in model.agents:
        for obj in agent.vocabulary:
            total_count += len(agent.vocabulary[obj])
    return total_count

def num_unique_words(model):
    unique_words = set()
    for agent in model.agents:
        for obj in agent.vocabulary:
            unique_words.update(agent.vocabulary[obj])
    return len(unique_words)

class PlayerAgent(mesa.Agent):
    def __init__(self, model: mesa.Model, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)
        
        self.vocabulary = dict()

    def generate_new_word(self):
        new_word = "".join(random.choices(string.ascii_letters, k=WORD_LENGTH))

        return new_word

    def speak(self,context):
        current_object = random.choice(context)
        if current_object not in self.vocabulary:
            new_word = self.generate_new_word()
            self.vocabulary[current_object] = [new_word]
            return (current_object, new_word)
        else:
            return (current_object, random.choice(self.vocabulary[current_object]))

    def hear(self,word,current_object):
        #If the agent doesn't have any words in the vocabulary associated with the object
        if current_object not in self.vocabulary:
            self.vocabulary[current_object] = [word]
            return False
        #If the agent already has words in its vocabulary for the object, but none of them match
        elif word not in self.vocabulary[current_object]:
            self.vocabulary[current_object].append(word)
            return False
        # If one of the words in the vocabulary matches the heard word
        else:
            self.vocabulary[current_object] = [word]
            return True


class Object():
    pass

class NamingGame(mesa.Model):
    def __init__(self, n_players: int, n_rounds: int, n_objects: int = 1):
        """This implements the naming game with a variable number of agents, rounds and objects to name"""
        super().__init__()
        self.n_agents = n_players
        self.n_rounds = n_rounds
        self.n_objects = n_objects
        self.objects = []
        self.interaction_results = []

        for _ in range(n_objects):
            self.objects.append(Object())

        for _ in range(self.n_agents):
          PlayerAgent(self)

        self.datacollector = mesa.DataCollector(
            {
                "all_words": num_total_words,
                "unique_words": num_unique_words,
                "success": self.get_interaction_results
            }
        )

    def step(self):
        speaker, hearer = self.agents.select(at_most=2)
        curr_object, word = speaker.speak(self.objects)
        success = hearer.hear(word, curr_object)
        self.interaction_results.append(success)
        self.datacollector.collect(self)
        # if num_unique_words(self) == len(self.objects) and num_total_words(self) == 
        self.running = True if self.n_rounds == None or self.steps < self.n_rounds else False


    def get_interaction_results(self):
        return self.interaction_results
        
        


if __name__ == "__main__":
    model = NamingGame(1000, 100, 1)
    model.run_model()
