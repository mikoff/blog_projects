import numpy as np
import matplotlib.pylab as plt

from matplotlib import animation

from ekf_slam.ekf_slam import EkfSlam
from ekf_slam.robot import Robot
from ekf_slam.map import generateLandmarkArray
from ekf_slam.animation import EkfSlamAnimation

np.set_printoptions(edgeitems=30, linewidth=1000, formatter={'float': '{: 0.4f}'.format})

VEL_STD_NOISE, OMEGA_STD_NOISE = 0.1, np.deg2rad(0.5)
OMEGA_BIAS_STD_NOISE = 0.005
DISTANCE_STD_NOISE, BEARING_STD_NOISE = 1.0, np.deg2rad(5.0)
MAX_RANGING_DISTANCE = 30.0
START_X, START_Y, START_THETA = -30.0, -30.0, 0.0

landmarks = generateLandmarkArray([
    [-40, -40], [-20, -20], [-20, -30],
    [0, -30], [15, -30], [30, -30],
    [30.0, -5.0], [45, -5.0], [45, -20], [30, -20],
    [20, 20], [30, 30], [40, 40], [50, 50.],
    [-45, 0], [-35, 10], [-25, 0], 
    [-50, 50], [-40, 40], [-30, 50], [-20, 40]])

np.random.seed(20)

ekfSlam = EkfSlam(START_X, START_Y, START_THETA, 
    len(landmarks), VEL_STD_NOISE, OMEGA_STD_NOISE, OMEGA_BIAS_STD_NOISE, DISTANCE_STD_NOISE, BEARING_STD_NOISE)
robot = Robot(START_X, START_Y, START_THETA, 
    VEL_STD_NOISE, OMEGA_STD_NOISE, MAX_RANGING_DISTANCE, DISTANCE_STD_NOISE, BEARING_STD_NOISE)

plt.ioff()
fig, ax = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [1, 1.18]})
ekfSlamAnimation = EkfSlamAnimation(fig, ax, landmarks, robot, ekfSlam)
anim = animation.FuncAnimation(fig, ekfSlamAnimation, init_func = ekfSlamAnimation.init,
                     frames = 3600, interval = 25, blit = True, repeat=True)

SAVE_TO_FILE_FLAG = False

if SAVE_TO_FILE_FLAG:
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['animation.embed_limit'] = 2**128
    FFwriter = animation.FFMpegWriter(fps=30, codec="libx264")     
    anim.save('ekf_slam_animation.mp4', writer = FFwriter )
else:
    plt.show()
