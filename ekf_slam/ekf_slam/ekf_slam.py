import numpy as np

def wrapToPi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class EkfSlam():
    def __init__(self, startX, startY, startTheta,
            nLandmarks, vel_std_noise, omega_std_noise, omega_bias_std_noise, distance_std_noise, bearing_std_noise):
        self.N_pose = 4
        self.N = self.N_pose + 2 * nLandmarks
        self.F_x = np.eye(self.N)
        self.F_x_pose = np.eye(self.N_pose)
        self.G_x = np.zeros((self.N_pose, self.N_pose - 1))

        self.Mu = np.zeros((self.N, 1))
        self.Mu[0, 0] = startX
        self.Mu[1, 0] = startY
        self.Mu[2, 0] = startTheta
        self.Mu[self.N_pose:, 0] = np.nan

        self.Sigma = np.zeros((self.N, self.N))
        self.Sigma[self.N_pose:, self.N_pose:] = np.eye(2 * nLandmarks, 2 * nLandmarks) * 10000

        self.vel_std_noise = vel_std_noise
        self.omega_std_noise = omega_std_noise
        self.omega_bias_std_noise = omega_bias_std_noise
        self.controlNoise = np.diag([vel_std_noise**2, omega_std_noise**2, omega_bias_std_noise**2])

        self.distance_std_noise = distance_std_noise
        self.bearing_std_noise = bearing_std_noise

    def predict(self, dt, velocity, omega):
        theta, omega_bias = self.Mu[2, 0], self.Mu[3, 0]
        sinTheta, cosTheta = np.sin(theta), np.cos(theta)

        omega = omega - omega_bias

        # move mean, or pose estimate
        self.Mu[0, 0] += velocity * cosTheta * dt
        self.Mu[1, 0] += velocity * sinTheta * dt
        self.Mu[2, 0] += omega * dt

        # move covariance: only pose covariances changes
        # update F_x
        self.F_x_pose[0, 2] = velocity * -sinTheta * dt
        self.F_x_pose[1, 2] = velocity * cosTheta * dt
        self.F_x_pose[2, 3] = -dt
        self.F_x[0:self.N_pose, 0:self.N_pose] = self.F_x_pose
        # update G_x
        self.G_x[0, 0] = cosTheta * dt
        self.G_x[1, 0] = sinTheta * dt
        self.G_x[2, 1] = dt
        self.G_x[3, 2] = 1.0
        # update Covariance
        self.Sigma = self.F_x @ self.Sigma @ self.F_x.T
        self.Sigma[0:self.N_pose, 0:self.N_pose] += self.G_x @ self.controlNoise @ self.G_x.T
        # self.Sigma[self.N_pose:, self.N_pose:] += np.eye(self.N - self.N_pose) * 0.1

    def correct(self, distances, bearings):
        # perform iterated update
        for idx, (distance, bearing) in enumerate(zip(distances, bearings)):
            if np.isnan(distance) or np.isnan(bearing):
                continue

            landmark_x_idx, landmark_y_idx = self.N_pose + 2 * idx, self.N_pose + 2 * idx + 1
            muX, muY, muTheta = self.Mu[0, 0], self.Mu[1, 0], self.Mu[2, 0]
            muLandmarkX, muLandmarkY = self.Mu[landmark_x_idx, 0], self.Mu[landmark_y_idx, 0]
            
            # if the landmark was not seen before, then initialize its position from measurement and current state
            if np.isnan(self.Mu[landmark_x_idx, 0]) or np.isnan(self.Mu[landmark_y_idx, 0]):
                self.Mu[landmark_x_idx, 0] = muLandmarkX = muX + distance * np.cos(bearing + muTheta)
                self.Mu[landmark_y_idx, 0] = muLandmarkY = muY + distance * np.sin(bearing + muTheta)

            dx, dy = muLandmarkX - muX, muLandmarkY - muY
            squaredDistance = dx ** 2 + dy ** 2

            # calculate expected measurement for landmark
            hatDistance = np.sqrt(squaredDistance)
            hatBearing = np.arctan2(dy, dx) - muTheta

            residualDistance = distance - hatDistance
            residualBearing = np.arctan2(np.sin(bearing - hatBearing), np.cos(bearing - hatBearing))

            residual = np.array([[residualDistance], [residualBearing]])

            # Fill measurement matrix
            # for pose estimate
            H = np.zeros((2, self.N))
            H[0, 0] = - hatDistance * dx
            H[0, 1] = - hatDistance * dy
            H[0, landmark_x_idx] = hatDistance * dx
            H[0, landmark_y_idx] = hatDistance * dy
            # for landmark estimate
            H[1, 0] = dy
            H[1, 1] = -dx
            H[1, 2] = -squaredDistance
            H[1, landmark_x_idx] = -dy
            H[1, landmark_y_idx] = dx
            H /= squaredDistance

            R = np.diag([self.distance_std_noise**2, self.bearing_std_noise**2])
            K = self.Sigma @ H.T @ np.linalg.pinv(H @ self.Sigma @ H.T + R)

            self.Mu = self.Mu + K @ residual
            # self.Sigma = (np.eye(self.N) - K @ H) @ self.Sigma
            self.Sigma = (np.eye(self.N) - K @ H) @ self.Sigma @ (np.eye(self.N) - K @ H).T  + K @ R @ K.T
            # print(np.all(np.linalg.eigvals(self.Sigma) > 0))

    def getPoseEstimate(self):
        return self.Mu[0:3, :].flatten()

    def getLandmarksCoordinatesEstimates(self):
        return self.Mu[self.N_pose::2].flatten(), self.Mu[self.N_pose + 1::2].flatten()

    def getCovarianceMatrix(self):
        return self.Sigma.copy()
