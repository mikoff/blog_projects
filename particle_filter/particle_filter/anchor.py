import numpy as np

class Anchor():
    def __init__(self, x = None, y = None):
        self.x = x
        self.y = y
        
    def isValid(self):
        return not (self.x is None or self.y is None)
    
    def position(self):
        return np.array([self.x, self.y]).reshape((-1, 1)) 
