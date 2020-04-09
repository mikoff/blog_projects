from uuid import uuid4
import numpy as np

def generateLandmarkArray(landmarkCoordinates):
    dtype = np.dtype([('id', object), ('x', np.float), ('y', np.float)])
    landmarks = np.zeros(len(landmarkCoordinates), dtype=dtype)
    for i, (x, y) in enumerate(landmarkCoordinates):
        landmarks[i]['id'] = uuid4().hex
        landmarks[i]['x'] = x
        landmarks[i]['y'] = y
    return landmarks

def wrapToPi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def R2D(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]])

# https://en.wikipedia.org/wiki/Kabsch_algorithm
def findRotationAndTranslation(A, B):
    idxs = np.all(np.isfinite(A), axis=0)
    centerA = np.mean(A[:, idxs], axis=1).reshape((2, 1))
    centerB = np.mean(B[:, idxs], axis=1).reshape((2, 1))
    A = A[:, idxs] - centerA
    B = B[:, idxs] - centerB

    C = A @ B.T
    U, D, V = np.linalg.svd(C)
    d = np.sign(np.linalg.det(C))
    R = V @ np.array([[1.0, 0.0], [0.0, d]]) @ U
    t = centerB - R @ centerA

    return R, t

# https://cookierobotics.com/007/
def getEllipsoidParametersFor2dCovariance(cov):
    (a, b), (_, c) = cov
    lambda1 = (a + c) * 0.5 + np.sqrt((0.5 * (a - c))**2 + b**2)
    lambda2 = (a + c) * 0.5 - np.sqrt((0.5 * (a - c))**2 + b**2)

    if lambda2 > lambda1:
        tmp = lambda1
        lambda1 = lambda2
        lambda2 = tmp

    if np.abs(b) < 1e-8 and a >= c:
        angle = 0
    elif np.abs(b) < 1e-8 and a < c:
        angle = np.pi / 2
    else:
        angle = np.arctan2(lambda1 - a, b)

    width = np.sqrt(lambda1) * 2 * 3
    height = np.sqrt(lambda2) * 2 * 3
    
    return width, height, angle

class RobotMover():
    def __init__(self, motionType, xlimits, ylimits):
        self.reminder = 200
        self.i = 0
        self.choices = np.random.randint(-6, 6, size=10)

        if motionType == "rectangle":
            self.mover = self.rectangleMover
        else:
            self.mover = self.randomMover

        self.xmin = xlimits[0]
        self.xmax = xlimits[1]
        self.ymin = ylimits[0]
        self.ymax = ylimits[1]

    def getControls(self, robotX, robotY):
        self.i += 1
        return self.mover(robotX, robotY)

    def rectangleMover(self, robotX, robotY):
        theta = 0.0
        if self.i % 300 == 0:
            theta = np.pi * 10 / 2

        return 0.1, 2.5, theta

    def randomMover(self, robotX, robotY):
        theta = 0.0
        if robotX > self.xmax or robotX < self.xmin or robotY > self.ymax or robotY < self.ymin:
            theta = np.deg2rad(np.random.randint(110, 270))
            self.reminder = np.random.randint(100, 200)
            self.choices = [0]
        elif self.i % self.reminder == 0:
            self.choices = np.random.randint(-6, 6, size=10)
            self.reminder = np.random.randint(200, 400)

        theta = theta + 0.15 * np.random.choice(self.choices)

        return 0.1, 2.5, theta