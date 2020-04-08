from matplotlib import collections as mc
from matplotlib.colors import Normalize

import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np

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

class EkfSlamAnimation(object):
    def __init__(self, fig, ax, landmarksTruePositions, robot, ekfSlam, extendPlot = 30):
        self.fig = fig
        self.ax = ax[0]
        self.landmarks = landmarksTruePositions
        self.robot = robot
        self.ekfSlam = ekfSlam

        ax[0].set(title = "EKF State", xlabel = "$x$, meters", ylabel = "$y$, meters")
        ax[1].set(title = "Covariance matrix", xlabel = "$\Sigma$ column $j$", ylabel = "$\Sigma$ row $i$")

        self.ax.set(aspect = "equal", 
            xlim = (landmarksTruePositions['x'].min() - extendPlot, landmarksTruePositions['x'].max() + extendPlot), 
            ylim = (landmarksTruePositions['y'].min() - extendPlot, landmarksTruePositions['y'].max() + extendPlot) )

        self.landmarksTruePositionsScatter = self.ax.scatter(
            landmarksTruePositions['x'], landmarksTruePositions['y'],
            color='black', marker='x', label = 'true landmark positions', zorder=3, s=150)

        self.positionTrue = self.ax.quiver([0.0], [0.0], [1.0], [0.0], 
            pivot='mid', color='black', units='inches', scale=4.0, label='True position')

        self.positionEst = self.ax.quiver([0.0], [0.0], [1.0], [0.0], 
            pivot='mid', color='violet', units='inches', scale=4.0, label='Estimated position')

        self.distanceAndBearingLines = self.ax.add_collection(
            mc.LineCollection([[(0.0, 0.0), (0.0, 0.0)]], linewidths=2, alpha=0.7, linestyles='dashed',
                              colors = cm.jet(np.linspace(0, 1, len(self.landmarks)))))

        self.landmarksEstimatedPositionsScatter = self.ax.scatter(
            [], [],
            color='magenta', marker='o', label = 'estimated landmark positions')

        data = np.zeros(1000)
        data[:] = np.nan
        self.landmarkMeasurementsScatter = self.ax.scatter(
            data, data,
            marker='.', alpha = 0.4, zorder=0, s=20.0, edgecolor='none', c=np.zeros((1000, 4)),
        )
        
        self.ax.legend(loc = 'upper left')

        self.im = ax[1].imshow(self.ekfSlam.Sigma, interpolation='none', vmin=-1, vmax=1, cmap=plt.cm.PiYG)
        fig.colorbar(self.im, ax=ax[1], shrink=0.7)

        self.localcm = cm.jet(np.linspace(0, 1, len(self.landmarks)))
        self.localcm[:,3] = 0.4

        self.text = self.ax.text(self.ax.get_xlim()[0] + (self.ax.get_xlim()[0] + self.ax.get_xlim()[1]) * 0.5,
                            self.ax.get_ylim()[0] + (self.ax.get_ylim()[0] + self.ax.get_ylim()[1]) * 0.05, "", fontsize=15)

        ellipses = mc.EllipseCollection(self.landmarks['x'], self.landmarks['y'], [0.0] * len(self.landmarks), 
            offsets = np.vstack((self.landmarks['x'], self.landmarks['y'])).T, 
            transOffset = self.ax.transData, units='xy', facecolors='none', edgecolors=cm.jet(np.linspace(0, 1, len(self.landmarks))), 
            alpha=0.95)
        self.confidenceEllipses = self.ax.add_collection(ellipses)
        self.confidenceEllipses.set_facecolor('none')

        self.i = 0
        self.choices = [0]
        self.reminder = 20

        plt.tight_layout()
        
    def moveState(self):
        # theta = 0.0
        # if self.robot.x > self.ax.get_xlim()[1] or self.robot.y < self.ax.get_ylim()[0] \
        #     or self.robot.x < self.ax.get_xlim()[0] or self.robot.y > self.ax.get_ylim()[1]:
        #     theta = np.deg2rad(np.random.randint(110, 270))
        #     self.reminder = np.random.randint(50, 100)
        #     self.i = 0
        #     self.choices = [0]
        # elif self.i % self.reminder == 0:
        #     self.choices = np.random.randint(-6, 6, size=10)
        #     self.reminder = np.random.randint(20, 40)
        #     self.i = 0
        # self.i += 1
        # theta = theta + 0.15 * np.random.choice(self.choices)
        theta = 0.0
        self.i += 1
        if self.i % 300 == 0:
            theta = np.pi * 10 / 2

        dt, vMeasured, omegaMeasured = self.robot.move(0.1, 2.5, theta)
        self.ekfSlam.predict(dt, vMeasured, omegaMeasured)

        distancesMeasured = self.robot.measureDistancesToLandmarks(self.landmarks)
        bearingsMeasured = self.robot.measureBearingsToLandmarks(self.landmarks)
        self.ekfSlam.correct(distancesMeasured, bearingsMeasured)

        truePositionX, truePositionY, trueOri, _ = self.robot.getPos()
        estPositionX, estPositionY, estOri = self.ekfSlam.getPoseEstimate()

        return (truePositionX, truePositionY, trueOri), \
            (estPositionX, estPositionY, estOri), \
            (distancesMeasured, bearingsMeasured), \
            self.ekfSlam.getLandmarksCoordinatesEstimates()
            

    def init(self):
        return (self.positionTrue, self.positionEst, self.distanceAndBearingLines, self.landmarksEstimatedPositionsScatter, 
            self.landmarkMeasurementsScatter, self.im, self.text, self.confidenceEllipses, )

    def __call__(self, frameIdx):
        truePose, estPose, measurements, estimatedLandmarksCoordinates = self.moveState()
        truePoseX, truePoseY, trueTheta = truePose
        estPoseX, estPoseY, estTheta = estPose
        distanceMeasurements, bearingMeasurements = measurements
        estimatedLandmarkCoordinatesX, estimatedLandmarkCoordinatesY = estimatedLandmarksCoordinates

        fromPoints = np.vstack((estimatedLandmarkCoordinatesX, estimatedLandmarkCoordinatesY))
        toPoints = np.vstack((self.landmarks['x'], self.landmarks['y']))
        R, t = findRotationAndTranslation(fromPoints, toPoints)

        # rigid transformation to estimated robot pose: to make the points visually aligned
        estPoseX, estPoseY = (R @ np.array([[estPoseX], [estPoseY]]) + t).flatten()
        estTheta -= np.arccos(R[0, 0])

        self.positionTrue.set_offsets([truePoseX, truePoseY])
        self.positionTrue.set_UVC(np.cos(trueTheta), np.sin(trueTheta))

        self.positionEst.set_offsets([estPoseX, estPoseY])
        self.positionEst.set_UVC(np.cos(estTheta), np.sin(estTheta))

        measurementSegments = []
        measurementEndPoints = np.zeros((len(distanceMeasurements), 2))

        R2DTheta = R2D(estTheta)
        for i, (distance, bearing) in enumerate(zip(distanceMeasurements, bearingMeasurements)):
            msrSegmentWrtRobot = R2DTheta @ R2D(bearing) @ np.array([[distance], [0.0]])
            segEndX, segEndY = estPoseX + msrSegmentWrtRobot[0,0], estPoseY + msrSegmentWrtRobot[1,0]

            measurementSegments.append([(estPoseX, estPoseY), (segEndX, segEndY)])
            measurementEndPoints[i, :] = segEndX, segEndY

        self.distanceAndBearingLines.set_segments(measurementSegments)

        alignedLandmarks = R @ fromPoints + t
        self.landmarksEstimatedPositionsScatter.set_offsets(
            np.c_[alignedLandmarks[0, :], alignedLandmarks[1, :]])

        idxs = np.isfinite(distanceMeasurements)
        numValid = np.sum(idxs)
        print(idxs, numValid)
        self.landmarkMeasurementsScatter._offsets = np.roll(self.landmarkMeasurementsScatter._offsets, -numValid, 0)
        self.landmarkMeasurementsScatter._offsets[-numValid:, :] = measurementEndPoints[idxs]

        print(self.landmarkMeasurementsScatter._offsets.shape)
        print(self.landmarkMeasurementsScatter._facecolors.shape)
        self.landmarkMeasurementsScatter._facecolors = np.roll(self.landmarkMeasurementsScatter._facecolors, -numValid, 0)
        self.landmarkMeasurementsScatter._facecolors[-numValid:, :] = self.localcm[idxs]

        covarianceMatrix = self.ekfSlam.getCovarianceMatrix()
        covarianceMatrix[covarianceMatrix > 1000] = np.max(covarianceMatrix[covarianceMatrix < 1000])
        self.im.set_array(covarianceMatrix / np.max(covarianceMatrix))

        self.text.set_text("Frame: " + str(frameIdx))

        # for confidenceEllipse in self.confidenceEllipses:
        angles = np.zeros(len(self.landmarks))
        widths, heights = np.zeros(len(self.landmarks)), np.zeros(len(self.landmarks))
        for i in range(0, len(self.landmarks)):
            idxCovStart, idxCovEnd = self.ekfSlam.N_pose + 2 * i, self.ekfSlam.N_pose + 2 * i + 2
            (a, b), (_, c) = R @ self.ekfSlam.Sigma[idxCovStart:idxCovEnd, idxCovStart:idxCovEnd]
            lambda1 = (a + c) * 0.5 + np.sqrt((0.5 * (a - c))**2 + b**2)
            lambda2 = (a + c) * 0.5 - np.sqrt((0.5 * (a - c))**2 + b**2)
            if np.abs(b) < 1e-8 and a >= c:
                angle = 0
            elif np.abs(b) < 1e-8 and a < c:
                angle = np.pi / 2
            else:
                angle = np.arctan2(lambda1 - a, b)
            angles[i] = angle
            widths[i] = np.sqrt(lambda1) * 2 * 3
            heights[i] = np.sqrt(lambda2) * 2 * 3
            print(lambda1, lambda2, angle)

        self.confidenceEllipses._widths = widths
        self.confidenceEllipses._heights = heights
        self.confidenceEllipses._angles = angles
        self.confidenceEllipses.set_offsets(
            np.c_[alignedLandmarks[0, :], alignedLandmarks[1, :]])
        # self.confidenceEllipses.set_edgecolor('black')

        return (self.positionTrue, self.positionEst, self.distanceAndBearingLines, self.landmarksEstimatedPositionsScatter, 
            self.landmarkMeasurementsScatter, self.im, self.text, self.confidenceEllipses, )
