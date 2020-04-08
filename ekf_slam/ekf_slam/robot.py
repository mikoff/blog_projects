import numpy as np

def wrapToPi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class Robot():
    def __init__(self, x0, y0, theta0, v_std_noise, omega_std_noise, max_ranging_distance, distance_meas_std, bearing_meas_std):
        self.x = x0
        self.y = y0
        self.theta = theta0
        self.v_std_noise = v_std_noise
        self.omega_std_noise = omega_std_noise
        self.omega_bias = np.random.normal(0.0, self.omega_std_noise)
        self.omega_bias = 0.02
        self.max_ranging_distance = max_ranging_distance
        self.distance_meas_std = distance_meas_std
        self.bearing_meas_std = bearing_meas_std

        
    def move(self, dt, velocity, omega):
        self.x += velocity * np.cos(self.theta) * dt
        self.y += velocity * np.sin(self.theta) * dt
        self.theta += omega * dt

        # self.x += velocity / omega * (-np.sin(self.theta) + np.sin(self.theta + omega * dt))
        # self.y += velocity / omega * (np.cos(self.theta) - np.cos(self.theta + omega * dt))
        # self.theta += omega * dt

        return dt, velocity + np.random.normal(0, self.v_std_noise), omega + np.random.normal(0, self.omega_std_noise) + self.omega_bias
        
    def getPos(self):
        return self.x, self.y, self.theta, self.omega_bias

    def getTrueDistancesToLandmarks(self, landmarks):
        dx, dy = (self.x - landmarks['x']), self.y - landmarks['y']
        distancesToLandmarks = np.sqrt(dx**2 + dy**2)

        return distancesToLandmarks

    def measureDistancesToLandmarks(self, landmarks):
        trueDistancesToLandmarks = self.getTrueDistancesToLandmarks(landmarks)

        visibleLandmarksIdxs = trueDistancesToLandmarks < self.max_ranging_distance
        trueDistancesToLandmarks[np.logical_not(visibleLandmarksIdxs)] = np.nan
        
        return trueDistancesToLandmarks + np.random.normal(0.0, self.distance_meas_std, size=len(landmarks))

    def getTrueBearingsToLandmarks(self, landmarks):
        dx, dy = landmarks['x'] - self.x, landmarks['y'] - self.y

        return wrapToPi(np.arctan2(dy, dx) - self.theta)

    def measureBearingsToLandmarks(self, landmarks):
        trueBearings = self.getTrueBearingsToLandmarks(landmarks)

        trueDistancesToLandmarks = self.getTrueDistancesToLandmarks(landmarks)
        visibleLandmarksIdxs = trueDistancesToLandmarks < self.max_ranging_distance

        trueBearings[np.logical_not(visibleLandmarksIdxs)] = np.nan

        return wrapToPi(trueBearings + np.random.normal(0.0, self.bearing_meas_std, size=len(landmarks)))

