from pylab import *
from body import Body
from discval import DiscVal


class BackAndForthBody(Body) :

    def __init__(self, model) :
        allowed_sensor_values = np.linspace(0,1,20)
        allowed_motor_values = np.linspace(-0.5,0.5,20)

        os = DiscVal(allowed_sensor_values, 0, name = "OS")
        om = DiscVal(allowed_motor_values, 0, name = "OM")

        super().__init__(model, radius = 0.5, sensor_length=0.333, sensor_βs=[0], 
                         sensors = [os], motors = [om])
        self.α = 0.0

    def update_sensors(self):
        light_pos = 3.0
        dsq = (self.x-light_pos)**2 #+ self.y**2
        self.sensors[0].value = 1.0/(1.0+dsq)

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
        self.sms_familiarity_matrix *= 0.9999


    def training_phase(self):
        # t = self.model.it * self.DT *5
        # m = cos(t/2)/2
        # self.next_motors[0].value = m
        
        if self.model.it % 500 == 0 :
            self.x = np.random.randn()*5.0

        t = self.model.it * self.DT *5
        θ = 4.0*cos(t/2)/2
        m = (θ - self.x) * 0.5
        m = self.next_motors[0].clip_value(m)
        self.next_motors[0].value = m

