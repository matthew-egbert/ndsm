from body import Body
from experiments.plotters.pattern_publication_plots import pattern_publication_plots
from experiments.plotters.position_time_slices_plot import position_time_slices_plot
from light_body import LightBody
from pattern_body import PatternBody
from brain import Brain
from experiments.plotters.plotting_utils import percent_complete, running_average
import dill as pickle
from tracker import TrackedValue,TrackingManager
from experiments.experiment import Experiment
import numpy as np
from pylab import *

from world import EmptyWorld

class LightExperiment(Experiment):
    def __init__(self,model,name=None) :
        super().__init__(model,name)
        self.duration                = 256000
        self.training_stop_iteration = 0
        
        ## LIGHT EXPERIMENT
        self.model.TIMESERIES_LENGTH = 256
        self.Ω = 64
        self.β = 512

        self.model.world = EmptyWorld(self.model);
        self.model.body  = LightBody(self.model, DT=self.model.DT);
        self.model.brain = Brain(self.model,Ω=self.Ω,β=self.β)

        ## DATA TO TRACK
        self.tracker.add_pickle_obj('training_stop_iteration',self.training_stop_iteration)

        self.tracker.track('time','model.time',should_sample=self.EVERY_ITERATION)
        self.tracker.track('prediction_error','model.brain.prediction_error',should_sample=self.EVERY_ITERATION)
        self.tracker.track('x','model.body.x',should_sample=self.EVERY_ITERATION)
        self.tracker.track('y','model.body.y',should_sample=self.EVERY_ITERATION)
        self.tracker.track('α','model.body.α',should_sample=self.EVERY_ITERATION)
        self.tracker.track('sms','model.body.sms',should_sample=self.EVERY_ITERATION)

        self.add_default_trackers()

    def iterate(self) :
        if self.model.it % 100 == 0 :
            color = 'c'
            if self.model.body.TRAINING_PHASE :
                color = 'r'
            pe = self.model.brain.prediction_error
            if pe != 0 :
                pe = log(pe)
            percent_complete(self.model.it,self.duration,title=f'Light Exp. error exponent={pe:.3f}',color=color)
            
        self.tracker.iterate(self)
        if self.model.it > self.duration :
            self.end()

        if self.model.it > self.training_stop_iteration :
            self.model.body.TRAINING_PHASE = False





