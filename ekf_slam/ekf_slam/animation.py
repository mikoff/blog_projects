from matplotlib import collections as mc
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse

import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np

from ekf_slam.helpers import *

class EkfSlamAnimation(object):
    def __init__(self, fig, ax, landmarksTruePositions, robot, ekfSlam, extendPlot, controller):
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

        self.positionConfidenceEllipse = Ellipse((0.0, 0.0), 0.0, 0.0, 0.0, edgecolor='magenta', facecolor='none')
        self.ax.add_patch(self.positionConfidenceEllipse)

        self.i = 0

        self.controller = controller

        plt.tight_layout()
        
    def moveState(self):
        dt, uV, uTheta = self.controller.getControls(self.robot.x, self.robot.y)

        dt, vMeasured, omegaMeasured = self.robot.move(dt, uV, uTheta)
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
            self.landmarkMeasurementsScatter, self.im, self.text, self.confidenceEllipses, self.positionConfidenceEllipse, )

    def __call__(self, frameIdx):
        # print(frameIdx)
        truePose, estPose, measurements, estimatedLandmarksCoordinates = self.moveState()
        truePoseX, truePoseY, trueTheta = truePose
        estPoseX, estPoseY, estTheta = estPose
        distanceMeasurements, bearingMeasurements = measurements
        estimatedLandmarkCoordinatesX, estimatedLandmarkCoordinatesY = estimatedLandmarksCoordinates

        # fromPoints= np.vstack((estimatedLandmarkCoordinatesX, estimatedLandmarkCoordinatesY))
        # toPoints = np.vstack((self.landmarks['x'], self.landmarks['y']))
        # R, t = findRotationAndTranslation(fromPoints, toPoints)
        # alignedLandmarks = R @ fromPoints + t

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

        self.landmarksEstimatedPositionsScatter.set_offsets(np.c_[estimatedLandmarkCoordinatesX, estimatedLandmarkCoordinatesY])

        idxs = np.isfinite(distanceMeasurements)
        numValid = np.sum(idxs)
        if numValid > 0:
            self.landmarkMeasurementsScatter._offsets = np.roll(self.landmarkMeasurementsScatter._offsets, -numValid, 0)
            self.landmarkMeasurementsScatter._offsets[-numValid:, :] = measurementEndPoints[idxs]

            self.landmarkMeasurementsScatter._facecolors = np.roll(self.landmarkMeasurementsScatter._facecolors, -numValid, 0)
            self.landmarkMeasurementsScatter._facecolors[-numValid:, :] = self.localcm[idxs]

        covarianceMatrix = self.ekfSlam.getCovarianceMatrix()
        covarianceMatrix[covarianceMatrix > 1000] = np.max(covarianceMatrix[covarianceMatrix < 1000])
        self.im.set_array(covarianceMatrix / np.max(covarianceMatrix))

        self.text.set_text("Frame: " + str(frameIdx))

        # for confidenceEllipse in self.confidenceEllipses:
        for i in range(0, len(self.landmarks)):
            idxCovStart, idxCovEnd = self.ekfSlam.N_pose + 2 * i, self.ekfSlam.N_pose + 2 * i + 2
            w, h, a = getEllipsoidParametersFor2dCovariance(self.ekfSlam.Sigma[idxCovStart:idxCovEnd, idxCovStart:idxCovEnd])
            self.confidenceEllipses._widths[i] = w
            self.confidenceEllipses._heights[i] = h
            self.confidenceEllipses._angles[i] = np.rad2deg(a)
            self.confidenceEllipses._offsets[i, 0] = estimatedLandmarkCoordinatesX[i]
            self.confidenceEllipses._offsets[i, 1] = estimatedLandmarkCoordinatesY[i]

        w, h, a = getEllipsoidParametersFor2dCovariance(self.ekfSlam.Sigma[0:2, 0:2])
        self.positionConfidenceEllipse.width = w
        self.positionConfidenceEllipse.height = h
        self.positionConfidenceEllipse.angle = np.rad2deg(a)
        self.positionConfidenceEllipse.center = (estPoseX, estPoseY)

        return (self.positionTrue, self.positionEst, self.distanceAndBearingLines, self.landmarksEstimatedPositionsScatter, 
            self.landmarkMeasurementsScatter, self.im, self.text, self.confidenceEllipses, self.positionConfidenceEllipse, )
