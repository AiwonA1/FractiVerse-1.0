import numpy as np

class Unipixel:
    def __init__(self, identifier, position=None, knowledge=[]):
        self.id = identifier
        self.knowledge = set(knowledge)
        self.position = position if position is not None else np.random.rand(3)
        self.vectors = {}

    def __rshift__(self, new_knowledge):
        self.knowledge.add(new_knowledge)
        return self

    def __repr__(self):
        return f"Unipixel({self.id}, Position: {self.position}, Knowledge: {list(self.knowledge)})"
