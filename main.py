from time import sleep
from kivy import platform
from kivy.config import Config
from kivy.clock import Clock

from back_and_forth_experiment import BackAndForthExperiment
from back_and_forth_body import BackAndForthBody
from braitenberg_experiment import BraitenbergExperiment
from pattern_experiment import PatternExperiment

if platform == 'linux':
    ratio = 2.0
    w = 1920
    Config.set('graphics', 'width', str(int(w)))
    Config.set('graphics', 'height', str(int(w / 2)))
    Config.set('graphics', 'fullscreen', 'false')
    Config.set('graphics', 'maxfps', '0')
    Config.set('postproc', 'maxfps', '0')
    # Disable pause on minimize
    Config.set('kivy', 'pause_on_minimize', '0')
    # Disable pause when window is out of focus
    Config.set('kivy', 'pause_on_focus', '0')

Config.set('kivy', 'log_level', 'debug')  # Set the log level to 'debug'
from pattern_body import PatternBody
from braitenberg_body import BraitenbergBody
import numpy as np
from threading import Thread

from body import Body
from brain import Brain
from world import BraitenbergWorld, EmptyWorld
from experiment import Experiment

import torch
import cProfile

# from kivy.logger import Logger
# Logger.setLevel(LOG_LEVELS["debug"])
# Logger.info('title: This is a info message.')
# Logger.debug('title: This is a debug message.')

class Model():
    def __init__(self, headless = False, experiment_class = None, *args, **kwargs):
        self.paused = False
        self.seed = np.random.randint(0,1000)
        self.seed = 6
        print(f"SEED: {self.seed}")
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.it = 0
        self.time = 0
        self.DT = 0.01
        self.sleep_amount = 0.0
        self.headless = headless

        self.TIMESERIES_LENGTH = 1024

        self.recording_sms = False
        self.sms_recording_history = None

        self.experiment = experiment_class(self)
                    
        self.init_env_drawables()
        self.init_body_drawables()
        if not self.headless :
            def iterate(arg) :
                sleep(self.sleep_amount)

                if self.sleep_amount < 0.5 : ## PAUSE if at max delay
                    self.iterate()

            Clock.schedule_interval(iterate, 0.0)

        else :
            while True :
                self.iterate()

        #self.brain.train_on_file("sms_recording.npy")
        # self.thread = Thread(target=self.run_clock, daemon=True)
        # self.thread.start()
        #print('hi')


    def run_clock(self):        
        print("Clock event triggered"+str(self.mm))
        self.mm += 1
        

    def init_env_drawables(self) :
        self.wall_lines = np.array(self.world.walls, dtype=float).reshape(-1,2)

    def init_body_drawables(self) :
        pass
        
    def iterate(self):
        if not self.paused:
            # if not self.headless :
            #     if self.it % 1 == 0 :
            #         print(f'##### it: {self.it} ')            
            self.brain.prepare_to_iterate()
            self.body.prepare_to_iterate()
            self.world.prepare_to_iterate()

            self.brain.iterate()
            self.body.iterate()
            self.world.iterate()
            self.wall_lines = np.array(self.world.walls, dtype=float).reshape(-1,2)
            
            self.experiment.iterate()
            
            if self.recording_sms:
                if self.sms_recording_history is None :
                    self.sms_recording_history = []
                self.sms_recording_history.append(self.body.sms)                
            else :
                if self.sms_recording_history is not None :
                    self.sms_recording_history = np.array(self.sms_recording_history)
                    np.save('sms_recording.npy',self.sms_recording_history)
                    self.sms_recording_history = None

            self.it += 1
            self.time = self.it * self.DT
            

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("simple_example")
    #parser.add_argument("counter", help="An integer will be increased by 1 and printed.", type=int)
    parser.add_argument("--headless", help="Run in headless mode.", action="store_true")
    parser.add_argument("--experiment", help="Experiment to run.", type=str)
    args = parser.parse_args()
    
    if args.experiment == 'pattern':
        experiment = PatternExperiment
    elif args.experiment == 'back_and_forth':
        experiment = BackAndForthExperiment
    elif args.experiment == 'braitenberg':
        experiment = BraitenbergExperiment
    else :
        experiment = PatternExperiment

    if args.headless:
        ## HEADLESS
        m = Model(headless=True, experiment_class=experiment)        
    else:       
        ## HEADFUL
        from rvit.core import init_rvit
        m = Model(experiment_class=experiment)
        init_rvit(m,rvit_file='rvit.kv',window_size=(500,250))
