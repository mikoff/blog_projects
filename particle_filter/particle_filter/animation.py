from particle_filter.anchor import *
from particle_filter.robot import *
from particle_filter.particle import *
from particle_filter.filter import *

from matplotlib import collections  as mc
from matplotlib.colors import Normalize
import matplotlib.pylab as plt

class ParticleFilterAnimation(object):
    def __init__(self, fig, ax, configuration, anchors):
        self.fig = fig
        self.ax = ax
        self.conf = configuration
        self.anchors = anchors
        
        conf = self.conf
        self.robot = Robot(conf['robot_start_x'], conf['robot_start_y'], conf['robot_start_theta'])
        self.particleFilter = ParticleFilter(
            conf['MIN_X'], conf['MAX_X'], 
            conf['MIN_Y'], conf['MAX_Y'], 
            conf['PARTICLES_NUM'])
        
        self.ax.set(aspect = "equal", xlim = (conf['MIN_X'], conf['MAX_X']), ylim=(conf['MIN_Y'], conf['MAX_Y']))

        normcolor = plt.Normalize(0.0, 0.01)
        self.particlesScatter = self.ax.scatter(
            [p.x for p in self.particleFilter.particles], [p.y for p in self.particleFilter.particles], 
            c=np.arange(conf['PARTICLES_NUM']), marker='.', alpha=0.5, cmap=plt.cm.rainbow, norm=normcolor, s=0.5,
            label='particles')
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.rainbow), ax = ax)
        cb.set_label('Relative particle weight')
        
        self.anchorsScatter = self.ax.scatter([a.x for a in anchors], [a.y for a in anchors], 
                                              color='red', marker='x', label='anchors')
        
        lines_coordinates = [[(0.0, 0.0), (0.0, 0.0)]]
        lc = mc.LineCollection(lines_coordinates, linewidths=1)
        self.lines = self.ax.add_collection(lc)
        
        self.positionEst, = self.ax.plot([0.0], [0.0], marker='o', markersize=6, color="violet", alpha=0.9,
                                        label='Estimated position')
        self.positionTrue, = self.ax.plot([0.0], [0.0], marker='+', markersize=5, color="black", alpha=0.9,
                                        label='True position')
        
        self.text = ax.text(conf['MIN_X'] + (conf['MIN_X'] + conf['MAX_X']) * 0.5, 
                            conf['MIN_Y'] + (conf['MIN_Y'] + conf['MAX_Y']) * 0.05, "")


        centers = np.array([[a.x, a.y] for a in anchors])
        cl = mc.EllipseCollection([1.] * 9, [1.] * 9, [1.] * 9, 
            offsets=centers, transOffset=ax.transData, units='xy', facecolors='none', color='black', alpha=0.5)
        self.circles = ax.add_collection(cl)
        self.circles.set_facecolor('none')
        
        self.ax.legend(loc = 'upper left')
        
    def moveState(self, step):
        if step == 'predict':
            v = np.random.normal(self.conf['AVG_SPEED'], 0.2)
            theta = np.random.normal(self.conf['AVG_THETA'], 0.05)
            
            self.robot.move(self.conf['DT'], v, theta)
            self.particleFilter.predict(self.conf['DT'] * v, self.conf['DT'] * theta, 
                                        self.conf['MOTION_STD'], 
                                        self.conf['THETA_STD'])
            weights = self.particleFilter.weights
            
        lines = []
        measurements = {}
        if step == 'correct':
            z, visible_anchors = [], []
            for a in self.anchors:
                dx, dy = self.robot.x - a.x, self.robot.y - a.y
                true_distance = np.sqrt(dx**2 + dy**2)
                measurements[a] = 0.0
                if true_distance <= self.conf['MAX_DISTANCE_MEASUREMENT_RANGE']:
                    meas = np.random.normal(true_distance, self.conf['DISTANCE_MEASUREMENT_STD'])
                    z.append(meas)
                    visible_anchors.append(a)
                    lines.append([(self.robot.x, self.robot.y), (a.x, a.y)])
                    measurements[a] = meas
                    
            self.particleFilter.correct(z, visible_anchors, self.conf['DISTANCE_MEASUREMENT_STD'])
            weights = self.particleFilter.weights
        
        if step == 'resample':
            if self.particleFilter.resamplingNeeded():
                self.particleFilter.resample()
                print("Resample")
            weights = self.particleFilter.weights

        x, y, th = self.particleFilter.getSolution()
        x_true, y_true, _ = self.robot.getPos()

        return self.particleFilter.particles, weights, lines, (x, y), measurements, (x_true, y_true)

    def init(self):
        return (self.particlesScatter, self.lines, self.positionEst, self.positionTrue, self.circles, self.text, )

    def __call__(self, i):
        print(i)
        if i % 3 == 0:
            step = "predict"
        if i % 3 == 1:
            step = "correct"
        if i % 3 == 2:
            step = "resample"
            
        particles, weights, lines_coordinates, (x, y), measurements, (x_true, y_true) = self.moveState(step)
        
        self.particlesScatter.set_offsets(np.c_[[p.x for p in particles], [p.y for p in particles]])
        normcolor = plt.Normalize(np.min(weights), np.max(weights))
        self.particlesScatter.set_array(normcolor(weights))
        self.lines.set_segments(lines_coordinates)
        self.positionEst.set_data(x, y)
        self.positionTrue.set_data(x_true, y_true)
        if step == 'correct':
            sizes = np.array([m for m in measurements.values()])
            self.circles._widths = np.asarray(sizes).ravel()
            self.circles._heights = np.asarray(sizes).ravel()
            self.circles.set_edgecolor('black')
        else:
            self.circles.set_edgecolor('none')
        self.text.set_text("STEP: " + step)

        return (self.particlesScatter, self.lines, self.positionEst, self.positionTrue, self.circles, self.text, ) 
