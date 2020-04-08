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