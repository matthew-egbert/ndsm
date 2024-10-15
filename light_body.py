from body import Body
from discval import DiscVal
from pylab import np

class LightBody(Body):
    """ Two motors and a light sensor. """
    def __init__(self, model, **kwargs) :
       
       allowed_sensor_values = np.linspace(0,1,9)
       allowed_motor_values = np.linspace(-2,2.0,2)

       os = DiscVal(allowed_sensor_values, 0, name = "OS")
       lm = DiscVal(allowed_motor_values, 0, name = "LM")
       rm = DiscVal(allowed_motor_values, 0, name = "RM")

       super().__init__(model, radius = 0.5, sensor_length=1.0, sensor_βs=[0],
                        sensors = [os], motors = [lm,rm], **kwargs)

    def training_phase(self):
        pass
    
    def update_sensors(self):
        """ By default, the sensor does nothing. """
        #print( np.sqrt(self.x**2 + self.y**2) / np.sqrt(50) )
        self.sensors[0].value = np.sqrt(self.x**2 + self.y**2) / np.sqrt(50)
        #print(self.sensors[0].value)

