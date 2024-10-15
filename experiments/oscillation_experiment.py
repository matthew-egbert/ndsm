import torch
from body import Body
from oscillation_body import OscillationBody
from discval import DiscVal, OneHotter
from pattern_body import PatternBody
from brain import Brain
from experiments.plotters.plotting_utils import arena_plot, percent_complete, running_average
import dill as pickle
from tracker import TrackedValue,TrackingManager 
from experiments.experiment import Experiment
import numpy as np
import networkx as nx
from pylab import *

from world import EmptyWorld, BraitenbergWorld
import matplotlib.patches as patches

class OscillationExperiment(Experiment):
    def __init__(self,model,name=None) :
        super().__init__(model,name)
        self.duration = 51200 #float('inf') # 10000
        self.training_stop_iteration = 25600

        ## BACK AND FORTH
        self.model.TIMESERIES_LENGTH = 1024        
        self.model.world = EmptyWorld(self.model); 
        self.model.body  = OscillationBody(self.model, DT=self.model.DT); 
        self.model.brain = Brain(self.model,Ω=128,β=512)
            
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
        self.model.brain.learning_rate_exponent = -3

        title = f'{self.name}'
        if self.model.body.TRAINING_PHASE :
            title += ' [TRAINING]'
        #title += f' lr:{self.model.brain.learning_rate}'
        title += f' err:{self.model.brain.prediction_error:.3f}'
        percent_complete(self.model.it,self.duration,title=title)
        self.tracker.iterate(self)
        
        if self.model.it > self.duration :
            self.end()
        
        if self.model.it > self.training_stop_iteration :
            self.model.body.TRAINING_PHASE = False
            #self.model.brain.ZERO_LEARNING_RATE = True
            #self.model.brain.learning_rate_exponent = -4

        # if self.model.it % 10000 == 0 :
        #     self.model.brain.image_2d_output()

