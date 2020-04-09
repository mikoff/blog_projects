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