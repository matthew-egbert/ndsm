import numpy as np
import torch

import os
from time import sleep

class Model():
    def __init__(self, seed, headless = False, experiment_class = None, *args, **kwargs):
        self.paused = False
        self.seed = np.random.randint(0,1000)
        self.seed = seed
        print(f"SEED: {self.seed}")
        self.OUTPUT_DIR = f'./results/{experiment_class.__name__}_{seed}/'
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.it = 0
        self.time = 0
        self.DT = 0.01
        self.sleep_amount = 0.0
        self.headless = headless

        self.TIMESERIES_LENGTH = 1024 ## how long a history of states is remembered

        self.recording_sms = False
        self.sms_recording_history = None

        self.experiment = experiment_class(self)

        self.init_env_drawables()
        self.init_body_drawables()


        if headless :
            while True :
                self.iterate()


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