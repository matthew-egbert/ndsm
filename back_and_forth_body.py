from pylab import *
from body import Body
from discval import DiscVal


class BackAndForthBody(Body) :

    def __init__(self, model, DT = 0.01) :
        # allowed_sensor_values = np.linspace(0,1,6)
        # allowed_motor_values = np.linspace(-0.75,0.75,4)
        allowed_sensor_values = np.linspace(0,1,21)
        allowed_motor_values = np.linspace(-0.75,0.75,21)
        

        os = DiscVal(allowed_sensor_values, 0, name = "OS")
        om = DiscVal(allowed_motor_values, 0, name = "OM")

        super().__init__(model, radius = 0.5, DT=DT,sensor_length=0.333, sensor_βs=[0], 
                         sensors = [os], motors = [om])
        self.α = 0.0
        self.x = 0.0

        self.Δ = 0

    def update_sensors(self):
        light_pos = 2.5
        dsq = (self.x-light_pos)**2

        noise = 0.0
        if np.random.rand() < 0.0 :
            noise = np.random.randn()*0.1
        v = 1.0/(1.0+dsq) + noise        
        self.sensors[0].value = self.sensors[0].clip_value(v)

    def update_position(self):
        k = 5.0       
        lm = rm = self.motors[0].value
        self.dx = cos(self.α)*(lm+rm) * k 
        self.dy = sin(self.α)*(lm+rm) * k 
        self.da = (rm-lm)*2.0*self.r * k 

        self.x += self.DT * self.dx
        self.y += self.DT * self.dy
        self.α += self.DT * self.da

    def set_motors(self,ms) :
        only_m = ms
        assert(only_m[0] < self.N_ALLOWED_MOTOR_VALUES)
        assert(only_m[0] >= 0)

        self.ms = [self.ALLOWED_MOTOR_VALUES[only_m[0]]]

    def iterate(self) :
        super().iterate()
        self.sms_familiarity_matrix[self.sensors[0].index,self.motors[0].index] += 0.1
        self.sms_familiarity_matrix *= 0.99

    def training_phase(self):
        ### time-based training
        t = self.model.it * self.DT
        θ = (2.25+0.25*sin(t/20)) * cos(2.0*t)
        m = (θ - self.x) * 1.0
        m = self.next_motors[0].clip_value(m)
        self.next_motors[0].value = m

        # ## CONDITIONAL TRAINING
        # sv = self.sensors[0].value
        # mv = self.motors[0].value
        # si = self.sensors[0].index
        # mi = self.motors[0].index

        # self.Δ += 1
        # δm = 0
        # if self.Δ > 5 :
        #     if si in [0] :
        #         δm = 1
        #     elif si in [5] :
        #         δm = -1
        #     self.Δ = 0 
        
        # new_mi = self.motors[0].clip_index(mi + δm)
        # self.next_motors[0].index = new_mi

