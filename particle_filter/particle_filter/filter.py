import numpy as np
from scipy.stats import norm
from copy import deepcopy

from particle_filter.particle import Particle

class ParticleFilter():
    def __init__(self, minX, maxX, minY, maxY, numParticles=1000):
        self.minX = minX
        self.maxX = maxX
        self.minY = minY
        self.maxY = maxY
        self.particles = [Particle.sample(minX, maxX, minY, maxY) for _ in range(0, numParticles)]
        self.weights = [1.0 / len(self.particles) for _ in range(0, numParticles)]
        
    def normalizeWeights(self):
        weightsSum = np.sum(self.weights)
        if weightsSum > 1e-10:
            self.weights = self.weights / weightsSum

    def getSolution(self):
        x, y, theta_x, theta_y = 0.0, 0.0, 0.0, 0.0
        
        for p, w in zip(self.particles, self.weights):
            x += w * p.x
            y += w * p.y
            theta_x += w * np.cos(p.theta)
            theta_y += w * np.sin(p.theta)
        return x, y, np.arctan2(theta_y, theta_x)
        
    def predict(self, distance, angle, distanceStd, thetaStd):
        distances = distance + np.random.normal(0.0, distanceStd, size = len(self.particles))
        for i, p in enumerate(self.particles):
            dx = distances[i]
            self.particles[i].x += dx * np.cos(p.theta)
            self.particles[i].y += dx * np.sin(p.theta)
            self.particles[i].theta += (angle + np.random.normal(0.0, thetaStd))
            
    def correct(self, measured_distances, anchors, measurement_std):
        for distance_measured, anchor in zip(measured_distances, anchors):
            for i, (p, w) in enumerate(zip(self.particles, self.weights)):
                distance_predicted = np.sqrt((anchor.x - p.x)**2 + (anchor.y - p.y)**2)
                distance_difference = np.abs(distance_measured - distance_predicted)
                self.weights[i] *= norm.pdf(0.0, distance_difference, measurement_std)
            self.normalizeWeights()
            
    def resample(self):
        r = np.random.uniform(0.0, 1.0 / len(self.particles))
        c = self.weights[0]
        i = 0
        resampled_particles = []
        for m in range(0, len(self.particles)):
            U = r + m / len(self.particles)
            while c < U:
                i += 1
                c += self.weights[i]
            resampled_particles += [deepcopy(self.particles[i])]
            
        self.particles = resampled_particles
        self.weights[:] = 1.0 / len(self.particles)

    def resamplingNeeded(self):
        weightsSquaredSum = np.sum(self.weights**2)
        if weightsSquaredSum < 1e-10:
            return True
        return 1.0 / weightsSquaredSum  < 0.333 * len(self.particles)
