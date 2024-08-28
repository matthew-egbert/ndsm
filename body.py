from pylab import *
from scipy.spatial.distance import cdist 
from SweepIntersectorLib.SweepIntersector import SweepIntersector
from discval import DiscVal, OneHotter
import copy

LEFT = 0
RIGHT = 1

class Body(object) :
    def __init__(self, model, DT = 0.01, radius = 1.0, motor_bias=0.0, sensor_βs = [], sensor_length = 3.3333,
                 sensors : list[DiscVal] = [], motors : list[DiscVal] = []) :
        self.model : Model = model
        self.x = 0.5
        self.y = 0.5
        self.α = 0.1
        self.r = radius
        self.motor_bias = motor_bias
        self.DT = DT
        self.TRAINING_PHASE = True

        self.sensors : list[DiscVal] = sensors
        self.motors : list[DiscVal] = motors
        self.next_motors : list[DiscVal] = [DiscVal(m.allowed_values,0) for m in motors]

        self.sensor_βs = np.array(sensor_βs)
        self.sensor_length = sensor_length

        self.dvs = sensors + motors
        self.onehotter = OneHotter(copy.deepcopy(self.dvs))

        self.h_length = self.model.TIMESERIES_LENGTH
        self.sms_h = np.ones((len(self.sensors)+len(self.motors),self.h_length))*0.0

        ## drawables
        self.s_h = np.zeros((len(self.sensors),self.h_length)) ## sensor_history ## TODO: can I have s_h and m_h be views of sms_h?
        self.m_h = np.zeros((len(self.motors),self.h_length))
        self.x_h = np.zeros(self.model.TRAIL_LENGTH)
        self.y_h = np.zeros(self.model.TRAIL_LENGTH)
        self.drawable_sensor_lines = np.zeros((len(self.sensors)*2,2))
        # self.sms_familiarity_matrix = np.zeros((len(self.motors,
        #                                         self.N_ALLOWED_SENSOR_VALUES))
        self.update_sensors()

                
    def update_sensors(self):
        ## sensor proximal coords
        spxs = self.x + np.cos(self.α+self.sensor_βs) * self.r
        spys = self.y + np.sin(self.α+self.sensor_βs) * self.r
        ## sensor distal coords
        sdxs = self.x + np.cos(self.α+self.sensor_βs) * (self.r+self.sensor_length) 
        sdys = self.y + np.sin(self.α+self.sensor_βs) * (self.r+self.sensor_length)

        self.sensor_line_segments = [((xp,yp),(xd,yd)) for xp,yp,xd,yd in zip(spxs,spys,sdxs,sdys)]
        self.drawable_sensor_lines = np.array(self.sensor_line_segments).reshape(-1,2)

        ## line segments to check for intersections
        line_segs = self.model.world.walls + self.sensor_line_segments
        # compute intersections        
        self.isector = SweepIntersector()   
        intersections = self.isector.findIntersections(line_segs)

        for sensor_index, sensor_seg in enumerate(self.sensor_line_segments) :
            intersection_locations = np.array([loc for loc in intersections[sensor_seg][1:-1]]).reshape(-1,2)
            distances = cdist(np.array(sensor_seg[0]).reshape(-1,2),intersection_locations)[0]
            if(len(distances) > 0 ):
                closest = argmin(distances)                        
                sensor_excitation = 1.0-(distances[closest] / self.sensor_length)
            else :
                sensor_excitation = 0.0
            
            self.sensors[sensor_index].value = sensor_excitation

    def prepare_to_iterate(self):
        if self.TRAINING_PHASE :                        
            self.training_phase()            
        else :
            self.onehotter.values = self.model.brain.actual_nn_output
            for ind in range(len(self.next_motors)) :    
                self.next_motors[ind].value = self.onehotter.values[len(self.sensors)+ind]
            print(f'actual nn output: {self.model.brain.actual_nn_output}')
            print(f'next motors: {[m.value for m in self.next_motors]}')

    def iterate(self) :
        for m,next_m in zip(self.motors,self.next_motors) :
            m.value = next_m.value

        self.sms = [s.value for s in self.sensors] + [m.value for m in self.motors]
        self.sms_h[:,self.model.it%self.h_length] = self.sms

        self.update_position()

        ## wrap
        ω = self.model.world.r # world radius
        if self.x > ω : 
            self.x -= 2.0*ω
        if self.y > ω : 
            self.y -= 2.0*ω

        if self.x < -ω : 
            self.x += 2.0*ω
        if self.y < -ω : 
            self.y += 2.0*ω

        self.update_sensors()

        self.x_h[self.model.it % self.model.TRAIL_LENGTH] = self.x
        self.y_h[self.model.it % self.model.TRAIL_LENGTH] = self.y

    def update_position(self):
        k = 5.0
        
        lm = self.motors[0].value 
        rm = self.motors[1].value

        if lm == rm :
            lm += self.motor_bias

        self.dx = cos(self.α)*(lm+rm) * k 
        self.dy = sin(self.α)*(lm+rm) * k 
        self.da = (rm-lm)*2.0*self.r * k 

        self.x += self.DT * self.dx
        self.y += self.DT * self.dy
        self.α += self.DT * self.da

    def randomize_position(self) :
        self.x = np.random.randn()*5.0
        self.y = np.random.randn()*5.0
        self.α = np.random.rand()*2.0*np.pi

if __name__ == '__main__' :
    from main import Model
    r = Body(Model())
    r.debug_plot()
    show()