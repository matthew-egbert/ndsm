from experiments.analysis import analyse
from tracker import TrackedValue,TrackingManager
import numpy as np
from pylab import *


class Experiment(object):
    def __init__(self,model,name=None) :
        self.model = model
        
        if name is None :
            self.name = type(self).__name__ ## gets the class name of the experiment by default
        else :
            self.name = name
    
        self.tracker = TrackingManager(self.model.OUTPUT_DIR)
        self.EVERY_ITERATION = lambda exp: True
        self.FREQUENTLY      = lambda exp: (m.it % 10) == 0  # type: ignore
        self.START           = lambda exp: exp.model.it == 0
        self.END             = lambda exp: exp.model.it == exp.duration-1

        self.tracker.add_pickle_obj('experiment',self.name)


    def add_default_trackers(self) :
        self.tracker.add_pickle_obj('seed',self.model.seed)
        self.tracker.add_pickle_obj('DT',self.model.DT)
        self.tracker.add_pickle_obj('Ω',self.model.brain.Ω)        
        self.tracker.add_pickle_obj('β',self.model.brain.β)
        self.tracker.add_pickle_obj('TIMESERIES_LENGTH',self.model.TIMESERIES_LENGTH)                                            
        self.tracker.add_pickle_obj('NN_PROPERTIES',str(self.model.brain.ffnn))

        #self.tracker.track('time','model.it',should_sample=self.EVERY_ITERATION)
        #self.tracker.track('x','model.body.x',should_sample=self.EVERY_ITERATION)
        #self.tracker.track('deltas','model.brain.deltas[:,:,0]',should_sample=self.EVERY_ITERATION)

    def reset(self) :
        pass

    def iterate(self) :
        self.tracker.iterate(self)
        if self.model.it > self.duration :
            self.end()

    def end(self) :
        print('Experiment completed.')
        self.tracker.save()
        analyse(self.model.OUTPUT_DIR)
        if self.model.headless :
            quit()
