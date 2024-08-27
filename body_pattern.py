from body import Body
from pylab import concatenate, np

from discval import DiscVal

class PatternBody(Body) :
    def __init__(self, model) :
        allowed_sensor_values = np.linspace(0,1,1)
        allowed_motor_values = np.linspace(-1,1.0,5)

        os = DiscVal(allowed_sensor_values, 0, name = "OS")
        lm = DiscVal(allowed_motor_values, 0, name = "LM")
        rm = DiscVal(allowed_motor_values, 0, name = "RM")

        super().__init__(model, radius = 0.5, sensor_length=1.0, sensor_βs=[0], 
                         sensors = [os], motors = [lm,rm])

    def training_phase(self):
        z = 0.0
        f = self.motors[0].max_value
        b = self.motors[0].min_value

        moves = [(f,f),(b,b),(b,f),(f,b)]
        motors = moves[self.model.it % 4]

        self.next_motors[0].value = motors[0]
        self.next_motors[1].value = motors[1]