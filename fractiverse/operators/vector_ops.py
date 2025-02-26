import numpy as np

class FractiVector:
    def __init__(self, text, weight=1.0):
        self.text = text
        self.weight = weight
        self.vector = self.compute_vector(text)

    def compute_vector(self, text):
        np.random.seed(hash(text) % (2**32))
        return np.random.rand(3) * self.weight

    def __add__(self, other):
        new_vector = (self.vector + other.vector) / 2
        return FractiVector(f"{self.text} + {other.text}", weight=(self.weight + other.weight) / 2)

    def __repr__(self):
        return f"FractiVector('{self.text}', vector={self.vector})"
