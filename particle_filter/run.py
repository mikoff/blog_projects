from matplotlib import animation

from particle_filter.anchor import *
from particle_filter.robot import *
from particle_filter.particle import *
from particle_filter.filter import *
from particle_filter.animation import *

anchors = [Anchor(5.0, 5.0),
           Anchor(10.0, 5.0),
           Anchor(10.0, 10.0),
           Anchor(5.0, 0.0), 
           Anchor(20.0, 10.0),
           Anchor(-5.0, 10.0),
           Anchor(-15.0, 0.0),
           Anchor(-20.0, -5.0),
           Anchor(-15.0, -10.0),
           Anchor(-20.0, 10.0)]

configuration = {'SIMULATION_TIME' : 10.0,
                 'MAX_DISTANCE_MEASUREMENT_RANGE' : 7.0,
                 'DT' : 0.5, 'AVG_SPEED' : 1.0, 'AVG_THETA' : 0.1,
                 'DISTANCE_MEASUREMENT_STD' : 1.0,
                 'MOTION_STD' : 0.5, 'THETA_STD' : 0.1,
                 'PARTICLES_NUM' : 500,
                 'MIN_X' : -25, 'MAX_X' : 30.0, 'MIN_Y' : -5.0, 'MAX_Y' : 25.0,
                 'robot_start_x' : 2.0, 'robot_start_y' : 2.0, 'robot_start_theta' : 0.0}

np.random.seed(10)
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
liveParticleFilter = ParticleFilterAnimation(fig, ax, configuration, anchors)
anim = animation.FuncAnimation(fig, liveParticleFilter, init_func = liveParticleFilter.init,
                     frames = 900, interval = 120, blit = True, repeat=False)
plt.tight_layout()

SAVE_TO_FILE_FLAG = True

if SAVE_TO_FILE_FLAG:
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['animation.embed_limit'] = 2**128
    FFwriter = animation.FFMpegWriter(fps=30, codec="libx264")     
    anim.save('pfanimation.mp4', writer = FFwriter )
else:
    plt.show()
