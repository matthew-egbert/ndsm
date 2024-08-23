import functools

from pylab import *

class SMCodec() :
    def __init__(self, variable_values : list[list[float]]) -> None:
        print("CREATING SMCODEC")
        self.variable_values = variable_values
        self.lens = [len(vv) for vv in self.variable_values]
        # quit()

    def values_to_onehot(self, values : np.array) -> list :
        #print(values)
        for i,v in enumerate(values) :
            if v > self.variable_values[i][-1] :
                raise ValueError(f"Value {v} is greater than max allowed value {self.variable_values[i][-1]}. SMINDEX: {i}")
            if v < self.variable_values[i][0] :
                raise ValueError(f"Value {v} is less than max allowed value {self.variable_values[i][0]}")

        ons = [self.values_to_indices([v],i)[0] for i,v in enumerate(values)]

        combined_index = int(sum(ons * prod(self.lens[i+1:]) for i, ons in enumerate(ons)))
        onehot = np.zeros(prod(self.lens),dtype=int32)
        onehot[combined_index] = 1
        return tuple(onehot)
    
    def indices_to_onehot(self, indices : np.array) -> list :
        for i,v in enumerate(indices) :
            if v > len(self.variable_values[i]) :
                raise ValueError(f"Index {v} is greater than the number of items. SMINDEX: {i}")
            if v < 0 :
                raise ValueError(f"Index {v} is less than zero. SMINDEX: {i}")

        ons = indices

        combined_index = int(sum(ons * prod(self.lens[i+1:]) for i, ons in enumerate(ons)))
        onehot = np.zeros(prod(self.lens),dtype=int32)
        onehot[combined_index] = 1
        return tuple(onehot)

    def values_from_onehot(self,onehot) :
        combined_index = argmax(onehot)
        ons = []
        for i in range(len(self.lens)):
            ons.append(int(combined_index // prod(self.lens[i+1:])))
            combined_index %= int(prod(self.lens[i+1:]))

        values = (self.variable_values[i][ons[i]] for i in range(len(self.lens)))
        return values

    def index_to_values(self,index) :
        onehot = np.zeros(prod(self.lens),dtype=int32)
        onehot[index] = 1
        return self.values_from_onehot(tuple(onehot))

    def values_to_indices(self,values,var_index) :
        for v in values :
            if v > self.variable_values[var_index][-1] :
                raise ValueError(f"Value {v} is greater than max allowed value {self.variable_values[var_index][-1]}")
            if v < self.variable_values[var_index][0] :
                raise ValueError(f"Value {v} is less than max allowed value {self.variable_values[var_index][0]}")

        #return [np.digitize(v,self.variable_values[var_index],right=True) for i,v in enumerate(values)]
        bins = self.variable_values[var_index]
        centers = (bins[1:]+bins[:-1])/2
        res = np.array(np.digitize(values, centers))
        # print()
        # print(bins)
        # print(centers)
        # print(f'VALUES[{var_index}]: {values} => {res}')
        return res
    
    def all_values_to_indices(self,values) :
        res = [self.values_to_indices([v],i)[0] for i,v in enumerate(values)]
        return res

if __name__ == '__main__' :
    N_SENSOR_VALUES = 2
    N_MOTOR_VALUES = 5
    sv = linspace(-1,1,N_SENSOR_VALUES)
    mv = linspace(-1,1,N_MOTOR_VALUES)

    smc = SMCodec([sv,sv,mv,mv])

    example_values = (sv[0],sv[1],mv[0],mv[0])
    onehot = smc.to_onehot(example_values)

    # print(argmax(onehot))
    # print(smc.from_onehot(tuple(onehot)))
    refound = list(smc.values_from_onehot(onehot))
    print(f'{example_values} => {argmax(onehot)} => {refound}')

    for index in range(100) :
        print(f'{index}: {list(smc.index_to_values(index))}' )