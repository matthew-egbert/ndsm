from pylab import *
from scipy.spatial.distance import cdist 
from SweepIntersectorLib.SweepIntersector import SweepIntersector
from smcodec import SMCodec


LEFT = 0
RIGHT = 1


class Body(object) :
    def __init__(self, model, DT = 0.01, radius = 1.0, sensor_length = 3.333, β=np.pi/4, γ=0, 
                 allowed_motor_values = [0,1], allowed_sensor_values = [0,1], n_motors=2, n_sensors=2) :
        """ β : gap between L/R sensor clusters
            γ : radial width of each sensor cluster 
        """

        self.model : Model = model
        self.x = 0.5
        self.y = 0.5
        self.α = 0.1
        self.r = radius
        self.DT = DT
        self.TRAINING_PHASE = True

        self.n_sensors = n_sensors
        self.n_motors = n_motors

        ## 2D BRAITENBERG BEHAVIOUR
        self.N_ALLOWED_SENSOR_VALUES = len(allowed_sensor_values)
        self.N_ALLOWED_MOTOR_VALUES = len(allowed_motor_values)
        self.ALLOWED_SENSOR_VALUES = allowed_sensor_values
        self.ALLOWED_MOTOR_VALUES = allowed_motor_values

        self.smcodec = SMCodec([self.ALLOWED_SENSOR_VALUES,]*self.n_sensors + [self.ALLOWED_MOTOR_VALUES,]*self.n_motors)

        self.init_sensors(sensor_length, β, γ)


        self.h_length = self.model.TIMESERIES_LENGTH
        self.sms_h = np.ones((self.n_sensors+self.n_motors,self.h_length))*0.0

        self.s_h = np.zeros((self.n_sensors,self.h_length)) ## sensor_history ## TODO: can I have s_h and m_h be views of sms_h?
        self.m_h = np.zeros((self.n_motors,self.h_length))
        self.x_h = np.zeros(self.model.TRAIL_LENGTH)
        self.y_h = np.zeros(self.model.TRAIL_LENGTH)
        self.drawable_sensor_lines = np.zeros((self.n_sensors*2,2))

        self.sms_familiarity_matrix = np.zeros((self.N_ALLOWED_MOTOR_VALUES,
                                                self.N_ALLOWED_SENSOR_VALUES))

        self.update_sensors()

    def set_motors(self,ms) :
        lmi,rmi = ms 
        assert(lmi < self.N_ALLOWED_MOTOR_VALUES)
        assert(rmi < self.N_ALLOWED_MOTOR_VALUES)
        assert(lmi >= 0)
        assert(rmi >= 0)

        # self.LMV = lmi
        # self.RMV = rmi
        self.ms = [self.ALLOWED_MOTOR_VALUES[lmi],self.ALLOWED_MOTOR_VALUES[rmi]]

    def test_motor_mappings(self) :        
        for i in range(self.N_ALLOWED_SENSOR_VALUES) :
            for j in range(self.N_ALLOWED_SENSOR_VALUES) :
                for k in range(self.N_ALLOWED_MOTOR_VALUES) :                
                    for l in range(self.N_ALLOWED_MOTOR_VALUES) :                        
                        ls = self.ALLOWED_SENSOR_VALUES[i]
                        rs = self.ALLOWED_SENSOR_VALUES[j]                
                        lm = self.ALLOWED_MOTOR_VALUES[k]
                        rm = self.ALLOWED_MOTOR_VALUES[l]
                        oneshot = self.smcodec.to_onehot( (ls,rs,lm,rm) )
                        ls2,rs2,lm2, rm2 = self.smcodec.from_onehot(oneshot)
                        print(f'{ls} {rs} {lm} {rm} \t=> {argmax(oneshot)} \t=> {ls2} {rs2} {lm2} {rm2}')

                        assert(ls == ls2)
                        assert(rs == rs2)
                        assert(lm == lm2)
                        assert(rm == rm2)
        quit()
                
    def update_sensors(self):
        ## sensor proximal coords
        spxs = self.x + np.cos(self.α+self.sensors_βs) * self.r
        spys = self.y + np.sin(self.α+self.sensors_βs) * self.r
        ## sensor distal coords
        sdxs = self.x + np.cos(self.α+self.sensors_βs) * (self.r+self.sensor_length) 
        sdys = self.y + np.sin(self.α+self.sensors_βs) * (self.r+self.sensor_length)

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
            #print(f'{sensor_index}: {intersection_locations} ::: {distances}')
            if(len(distances) > 0 ):
                closest = argmin(distances)                        
                self.raw_sensor_excitations[sensor_index] = 1.0-(distances[closest] / self.sensor_length)
            else :
                self.raw_sensor_excitations[sensor_index] = 0.0
        
        self.sensor_excitations = self.smcodec.values_to_indices(self.raw_sensor_excitations,0)

        for x in self.sensor_excitations :
            assert(x < self.N_ALLOWED_SENSOR_VALUES)
            assert(x >= 0)

    def randomizePosition(self) :
        self.x = np.random.rand()
        self.y = np.random.rand()
        self.α = np.random.rand()*2.0*np.pi

    def prepare_to_iterate(self):
        ms = self.smcodec.values_to_indices(self.model.brain.next_motor_state,-1)

        if self.TRAINING_PHASE :                        
            ms = self.training_motor_output()            
    
        self.next_motor_state = ms

    def iterate(self) :
        #print(f'it: {self.model.it} recording SMS to sms_h[:,{self.model.it%self.h_length}]')
        self.set_motors(self.next_motor_state)
        self.sms_h[:,self.model.it%self.h_length] = np.concatenate([self.raw_sensor_excitations,self.ms])
        self.dx = 0.0
        self.dy = 0.0

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
        motor_bias = 0.25
        lm = self.ms[0] + motor_bias
        rm = self.ms[1]
        self.dx = cos(self.α)*(lm+rm) * k 
        self.dy = sin(self.α)*(lm+rm) * k 
        self.da = (rm-lm)*2.0*self.r * k 

        self.x += self.DT * self.dx
        self.y += self.DT * self.dy
        self.α += self.DT * self.da

class BraitenbergBody(Body) :
    def __init__(self, model) :

        allowed_sensor_values = np.linspace(0,1,5)
        allowed_motor_values = np.linspace(-1,1.0,5)

        super().__init__(model,DT = 0.01, radius = 0.5, sensor_length=1.0, β=np.pi/4, γ=0,
                         allowed_motor_values = allowed_motor_values, 
                         allowed_sensor_values = allowed_sensor_values)

    def init_sensors(self, sensor_length, β, γ):
        self.sensor_length = sensor_length
        sensors_per_side = self.n_sensors // 2
        self.raw_sensor_excitations = np.zeros(self.n_sensors)
        self.sensor_excitations = np.zeros(self.n_sensors)
        self.l_sensors_βs = np.linspace(-β-γ/2,-β+γ/2,sensors_per_side)
        self.r_sensors_βs = np.linspace(β-γ/2,β+γ/2,sensors_per_side)
        self.sensors_βs = concatenate([self.l_sensors_βs,self.r_sensors_βs])

    def training_motor_output(self):
        print(self.sensor_excitations)
        ## braitenberg vehicle (more sensor and motor states)
        lm = 0.25-0.75*self.sensor_excitations[LEFT] + 0.5*self.sensor_excitations[RIGHT]
        rm = 0.25-0.75*self.sensor_excitations[RIGHT] + 0.5*self.sensor_excitations[LEFT]           
        lm = np.clip(lm,min(self.ALLOWED_MOTOR_VALUES),max(self.ALLOWED_MOTOR_VALUES))
        rm = np.clip(rm,min(self.ALLOWED_MOTOR_VALUES),max(self.ALLOWED_MOTOR_VALUES))
        lm,rm = self.smcodec.values_to_indices([lm,rm],-1) 
        return lm,rm


class SimpleBody(Body) :
    def __init__(self, model) :

        allowed_sensor_values = np.linspace(0,1,5)
        allowed_motor_values = np.linspace(-0.5,0.5,5)

        super().__init__(model,DT = 0.01, radius = 1.0, sensor_length=0.333, β=np.pi/4, γ=0,
                         allowed_motor_values = allowed_motor_values, 
                         allowed_sensor_values = allowed_sensor_values,
                         n_sensors=1, 
                         n_motors=1)
        self.α = 0.0

    def init_sensors(self, sensor_length, β, γ):
        self.sensor_length = 0
        self.raw_sensor_excitations = np.zeros(self.n_sensors)
        self.sensor_excitations = np.zeros(self.n_sensors)
        self.sensors_βs = np.array([0.0])

    def update_sensors(self):        
        dsq = self.x**2 + self.y**2
        self.raw_sensor_excitations[0] = 1.0/(1.0+dsq)        
        discretized_sensor_excitations = self.smcodec.values_to_indices(self.raw_sensor_excitations,0)
        self.rounded_sensor_excitations = self.ALLOWED_SENSOR_VALUES[discretized_sensor_excitations]
        self.sensor_excitations = self.rounded_sensor_excitations
        print(self.raw_sensor_excitations, self.sensor_excitations)

    def update_position(self):
        k = 5.0       
        lm = rm = self.ms[0]
        self.dx = cos(self.α)*(lm+rm) * k 
        self.dy = sin(self.α)*(lm+rm) * k 
        self.da = (rm-lm)*2.0*self.r * k 

        self.x += self.DT * self.dx
        self.y += self.DT * self.dy
        self.α += self.DT * self.da

    def set_motors(self,ms) :
        ## TODO: this is not yet written, just makes the robot mpove forward
        only_m = ms
        assert(only_m < self.N_ALLOWED_MOTOR_VALUES)
        assert(only_m >= 0)

        self.ms = [self.ALLOWED_MOTOR_VALUES[only_m]]

    def training_motor_output(self):
        t = self.model.it * self.DT *5
        m = cos(t/2)/2
        dm = self.smcodec.values_to_indices([m],-1)[0]
        print(f'm: {m:0.2f} \t dm: {dm} \t --> \t {self.ALLOWED_MOTOR_VALUES[dm]}')
        return dm


if __name__ == '__main__' :
    from main import Model
    r = Body(Model())
    r.debug_plot()
    show()