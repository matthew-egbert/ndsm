from pylab import *

class DiscVal() :
    """ A variable that has a finite and enumerated (i.e. indexed) number of states."""
    def __init__(self, values, initial_state_index=0, name="Unnamed DValue") :
        self.name = name

        self.allowed_values = values ## must increase monotonically
        bins = np.array(self.allowed_values)
        self.digitize_centers = (bins[1:]+bins[:-1])/2

        self._index = initial_state_index
        self.max_value = self.allowed_values[-1]
        self.min_value = self.allowed_values[0]

    def clip_value(self, v) :
        return min(max(v, self.min_value), self.max_value)

    @property
    def index(self) :
        return self._index

    @index.setter
    def index(self, state) :
        self._index = state

    @property
    def value(self) :
        return self.allowed_values[self._index]
    
    @value.setter
    def value(self, new_value) :        
        if new_value < self.allowed_values[0] or new_value > self.allowed_values[-1] :
            raise ValueError(f"{self.name} :: value {new_value} is out of range. ")
        index = np.array(np.digitize(new_value, self.digitize_centers))
        self._index = index
        
    def __repr__(self) :
        return f'{self.name}: {self.value} [{self.index}/{len(self.allowed_values)}]'

class OneHotter():
    def __init__(self, discvals) :
        self._discvals = discvals ## only set in the constructor, keeps track of allowed values        
        self.values_per_dv = [len(dv.allowed_values) for dv in self._discvals]
        self.onehot_length = np.prod(self.values_per_dv)

    @property
    def values(self) :
        return [dv.value for dv in self._discvals]

    @values.setter
    def values(self, vs):
        for index, dv in enumerate(self._discvals):
            dv.value = vs[index]

    @property
    def indices(self) :
        return [dv.index for dv in self._discvals]

    @indices.setter
    def indices(self, values):
        for ind, dv in enumerate(self._discvals):
            #print(f'{dv.name} :: index: {values[index]}')
            dv.index = values[ind]

    @property
    def onehot(self) :
        ons = self._discvals
        combined_index = int(sum(dv.index * prod(self.values_per_dv[i+1:]) for i, dv in enumerate(ons)))
        onehot = np.zeros(self.onehot_length, dtype=int)
        onehot[combined_index] = 1
        return onehot
    
    @onehot.setter
    def onehot(self, onehot) :
        combined_index = argmax(onehot)
        ons = []
        for i in range(len(self.values_per_dv)):
            ons.append(int(combined_index // prod(self.values_per_dv[i+1:])))
            combined_index %= int(prod(self.values_per_dv[i+1:]))
        for i, dv in enumerate(self._discvals):
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
            encoder.values = [s, m]
            onehot = encoder.onehot

            decoder.onehot = onehot
            # bas = backagain[0].value
            # bam = backagain[1].value
            print(f'ls,rs: {sensor.value, motor.value}    \t encoded: {onehot}, \t decoded: {decoder.values}')
            #print(f'ls,rs: {sensor.index, motor.index}    \t encoded: {onehot}, \t decoded: {decoder.indices}')

            assert(sensor.value == decoder.values[0])
            assert(motor.value == decoder.values[1])
