from body import Body
from pylab import concatenate, np

from discval import DiscVal

class PatternBody(Body) :
    def __init__(self, model, pattern_length, **kwargs) :
        allowed_sensor_values = np.linspace(0,1,2)
        allowed_motor_values = np.linspace(-1,1.0,2)

        os = DiscVal(allowed_sensor_values, 0, name = "OS")
        lm = DiscVal(allowed_motor_values, 0, name = "LM")
        rm = DiscVal(allowed_motor_values, 0, name = "RM")

        super().__init__(model, radius = 0.5, sensor_length=1.0, sensor_βs=[0], 
                         sensors = [], motors = [lm,rm], **kwargs)
        
        self.pattern_length = pattern_length
        z = 0.0
        f = self.motors[0].max_value
        b = self.motors[0].min_value        
        F = (f,f)
        B = (b,b)
        L = (b,f)
        R = (f,b)
        self.moves = [(f,f),(f,f),(f,f),(b,b),(b,f),(f,b)]
        #self.pattern = [self.moves[np.random.randint(len(self.moves))] for idx in range(pattern_length)]
        self.pattern = []
        for idx in range(2) :
            self.pattern.extend([F,F,F,F,F,R,R,R,R,R,F,F,F,F,L,L])
        for idx in range(2) :
            self.pattern.extend([R,B,R,B,R,B,R,B,R,B,F,F,F,F,L,L])
        # for idx in range(4) :
        #     self.pattern.extend([F,R,F,R,F,R,F,R,F,R,F,R,F,R,F,R])
        # for idx in range(4) :
        #     self.pattern.extend([F,L,L,F,L,L,F,L,L,F,L,L,F,L,L,F])
        # print(len(self.pattern))
        # quit()

    def update_sensors(self):
        # β = np.pi/4
        # a = self.α % (2*np.pi)
        # if a < (np.pi/2 + β) and a > (np.pi/2 - β) :
        #     self.sensors[0].value = np.random.choice([0.0,1.0])
        # else :
        #     self.sensors[0].value = 0.0
        if len(self.sensors) > 0 :
            self.sensors[0].value = 0.0

    def training_phase(self):
        motors = self.pattern[self.model.it % self.pattern_length]
        self.next_motors[0].value = motors[0]
        self.next_motors[1].value = motors[1]