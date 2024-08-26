from pylab import *

class DiscVal() :
    """ A variable that has a finite and enumerated (i.e. indexed) number of states."""
    def __init__(self, values, initial_state_index=0, name="Unnamed DValue") :
        self.name = name

        self.values = values ## must increase monotonically
        bins = np.array(self.values)
        self.digitize_centers = (bins[1:]+bins[:-1])/2

        self._index = initial_state_index

    @property
    def index(self) :
        return self._index

    @index.setter
    def index(self, state) :
        self._index = state

    @property
    def value(self) :
        return self.values[self._index]
    
    @value.setter
    def value(self, new_value) :
        if new_value < self.values[0] or new_value > self.values[-1] :
            raise ValueError(f"{self.name} :: value {value} is out of range. ")
        index = np.array(np.digitize(new_value, self.digitize_centers))
        self._index = index
        
class OneHotter():
    def __init__(self, dvs) :
        self._dvs = dvs
        self.lens = [len(dv.values) for dv in self._dvs]
        self.prod = np.prod(self.lens)

    @property
    def dvs(self) :
        return self._dvs

    @dvs.setter
    def dvs(self, dvalues):
        self._dvs = dvalues
        self.lens = [len(dv.values) for dv in self._dvs]
        self.prod = np.prod(self.lens)

    @property
    def onehot(self) :
        ons = self.dvs
        combined_index = int(sum(dv.index * prod(self.lens[i+1:]) for i, dv in enumerate(ons)))
        onehot = np.zeros(self.prod, dtype=int)
        onehot[combined_index] = 1
        return onehot
    
    @onehot.setter
    def onehot(self, onehot) :
        combined_index = argmax(onehot)
        ons = []
        for i in range(len(self.lens)):
            ons.append(int(combined_index // prod(self.lens[i+1:])))
            combined_index %= int(prod(self.lens[i+1:]))
        for i, dv in enumerate(self.dvs):
            dv.index = ons[i]
    


if __name__ == '__main__':
    dv = DiscVal([0,10,20,30,40,50,60,70,80,90], 5, name = "LS")    

    dv.value = 24.9
    assert(dv.index == 2)
    assert(dv.value == 20)

    dv.value = 25.0
    assert(dv.index == 3)
    assert(dv.value == 30)
    
    dv.index = 7
    assert(dv.index == 7)
    assert(dv.value == 70)

    allowed_sensor_values = [0,0.5,1.0]
    allowed_motor_values = [0,0.333,0.666,1.0]
    sensor = DiscVal(allowed_sensor_values, 0, name = "LS")
    motor = DiscVal(allowed_motor_values, 0, name = "RS")
    
    encoder = OneHotter([sensor, motor])
    decoder = OneHotter([sensor, motor])
    for s in allowed_sensor_values:
        for m in allowed_motor_values:
            sensor.value = s
            motor.value = m
            encoder.dvs = [sensor, motor]
            onehot = encoder.onehot

            decoder.onehot = onehot
            backagain = decoder.dvs
            bas = backagain[0].value
            bam = backagain[1].value
            print(f'ls,rs: {sensor.value, motor.value}    \t encoded: {onehot}, \t decoded: {bas, bam}')

            assert(sensor.value == bas)
            assert(motor.value == bam)
