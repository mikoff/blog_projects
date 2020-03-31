import numpy as np

class Robot():
    def __init__(self, x0 = 0.0, y0 = 0.0, theta = 0.0):
        self.x = x0
        self.y = y0
        self.theta = theta
        
    def move(self, dt = 1.0, v = 1.0, theta_rate = 0.1):
        self.x += v * dt * np.cos(self.theta)
        self.y += v * dt * np.sin(self.theta)
        self.theta += dt * theta_rate
        
    def getPos(self):
        return self.x, self.y, self.theta
