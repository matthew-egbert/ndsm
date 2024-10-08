from rvit.tracker import TrackedValue,TrackingManager
import numpy as np
from pylab import *


class Experiment(object):
    def __init__(self,model,name=None) :
        self.model = model
        self.duration = float('inf') # 10000
        
        if name is None :
            self.name = type(self).__name__ ## gets the class name of the experiment by default
        else :
            self.name = name
    
        self.tracker = TrackingManager(self.name)
        self.EVERY_ITERATION = lambda exp: True
        self.FREQUENTLY      = lambda exp: (m.it % 10) == 0
        self.START           = lambda exp: exp.model.it == 0
        self.END             = lambda exp: exp.model.it == exp.duration-1

        self.tracker.add_pickle_obj('seed',self.model.seed)

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
        #self.tracker.save()
        #self.tracker.plot_x_against_y('time','x')
        # self.tracker.plot_x_against_y('time','ages')

        # time = self.tracker.data('time')
        # deltas = np.array(self.tracker.data('deltas'))
        # figure()
        # pl.plot(time,deltas[:,0,:],color='k')
        # pl.plot(time,deltas[:,1,:],color='r')
        # # for t,d1,d2 in zip(time,deltas[:,0,:],deltas[:,1,:]) :
        # #     plot([t,t],[d1,d2],color='k',alpha=0.2)
        # pl.xlabel('time')
        # pl.ylabel('deltas')
        # self.tracker.save_additional_figure('deltas')
        #quit()
