from kivy import platform
from kivy.config import Config
from kivy.clock import Clock

from body_back_and_forth import BackAndForthBody

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
from body_pattern import PatternBody
from body_braitenberg import BraitenbergBody
from rvit.core import init_rvit
import numpy as np
from threading import Thread

from body import Body
from brain import Brain
from world import World, EmptyWorld

# from kivy.logger import Logger
# Logger.setLevel(LOG_LEVELS["debug"])
# Logger.info('title: This is a info message.')
# Logger.debug('title: This is a debug message.')

class Model():
    def __init__(self, *args, **kwargs):
        self.paused = False
        self.it = 0
        self.draw_frequency = 1.0
        self.TIMESERIES_LENGTH = 256
        self.TRAIL_LENGTH = 256
        
        
        ## ## BRAITENBERG
        self.world : World = World(self); self.body : Body = BraitenbergBody(self); self.brain : Brain = Brain(self,sm_duration=50)
        
        ## ## PATTERN
        #self.world : World = EmptyWorld(self); self.body : Body = PatternBody(self); self.brain : Brain = Brain(self,sm_duration=50)
        
        ## ## BACK AND FORTH
        # self.world : World = EmptyWorld(self); self.body : Body = BackAndForthBody(self); self.brain : Brain = Brain(self,sm_duration=50)
        
        
        
        self.init_env_drawables()
        self.init_body_drawables()

        def iterate(arg):            
            self.iterate()

        Clock.schedule_interval(iterate, 0.0)
        self.mm = 0

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
            self.it += 1
            print(f'##### it: {self.it} ')
            self.brain.prepare_to_iterate()
            self.body.prepare_to_iterate()

            self.brain.iterate()
            self.body.iterate()
            
            if (self.it % int(self.draw_frequency)) == 100 :   
                self.updateDrawnArrays()
            
    def updateDrawnArrays(self):
        pass

if __name__ == '__main__':
    pass
    m = Model()
    init_rvit(m,rvit_file='rvit.kv',window_size=(500,250))
    # skivy.activate()
    # ModelApp().run()
