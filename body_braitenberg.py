from body import Body
from discval import DiscVal


from pylab import concatenate, np


class BraitenbergBody(Body) :
    def __init__(self, model) :

        allowed_sensor_values = np.linspace(0,1,3)
        allowed_motor_values = np.linspace(-1.0,1.0,3)
        ls = DiscVal(allowed_sensor_values, 0, name = "LS")
        rs = DiscVal(allowed_sensor_values, 0, name = "RS")
        lm = DiscVal(allowed_motor_values, 0, name = "LM")
        rm = DiscVal(allowed_motor_values, 0, name = "RM")

        super().__init__(model,DT = 0.01, radius = 0.5, sensor_length=1.0, sensor_βs=[-np.pi/4,np.pi/4], motor_bias=0.,
                         sensors = [ls,rs], motors = [lm,rm])

    def training_phase(self):
        ## braitenberg vehicle (more sensor and motor states)
        ls = self.sensors[0].value
        rs = self.sensors[1].value

        sensor_noise_scale = 0.1
        ls += np.random.randn()*sensor_noise_scale
        rs += np.random.randn()*sensor_noise_scale

        lm = 0.25-0.75*ls + 0.5*rs
        rm = 0.25-0.75*rs + 0.5*ls

        motor_noise_scale = 0.3
        lm += np.random.randn()*motor_noise_scale
        rm += np.random.randn()*motor_noise_scale

        lm = self.motors[0].clip_value(lm)
        rm = self.motors[1].clip_value(rm)

        self.next_motors[0].value = lm
        self.next_motors[1].value = rm

        # if self.model.it % 1000 == 0 :
        #     self.randomize_position()