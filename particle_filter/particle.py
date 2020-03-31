import numpy as np

class Particle():
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        
    def __str__(self):
        return f'Particle({self.x:.3f}, {self.y:.3f}, {np.rad2deg(self.theta):.3f})'
        
    @staticmethod
    def sample(minX, maxX, minY, maxY):
        uniform = np.random.uniform
        return Particle(uniform(minX, maxX), uniform(minY, maxY), uniform(0.0, 2 * np.pi)) 
