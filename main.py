from kivy import platform
from kivy.config import Config
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.clock import Clock
if platform == 'linux':
    ratio = 2.0
    w = 1920
    Config.set('graphics', 'width', str(int(w)))
    Config.set('graphics', 'height', str(int(w / 2)))
    #Config.set('graphics', 'fullscreen', 'auto')
Config.set('kivy', 'log_level', 'debug')  # Set the log level to 'debug'

from rvit.core import init_rvit

import cmath
import numpy as np

from body import Body
from brain import Brain
from world import World

# from kivy.logger import Logger
# Logger.setLevel(LOG_LEVELS["debug"])
# Logger.info('title: This is a info message.')
# Logger.debug('title: This is a debug message.')

class Model():
    def __init__(self, *args, **kwargs):
        self.it = 0
        self.draw_frequency = 1.0
        self.TIMESERIES_LENGTH = 256
        
        self.world : World = World(self)
        self.body : Body = Body(self)
        self.brain : Brain = Brain(self)
        
        self.init_env_drawables()
        self.init_body_drawables()

        def iterate(arg):
            self.iterate()

        Clock.schedule_interval(iterate, 1.0 / 500.0)

    def init_env_drawables(self) :
        self.wall_lines = np.array(self.world.walls, dtype=float).reshape(-1,2)

    def init_body_drawables(self) :
        pass
        
    def iterate(self):
        self.it += 1
        self.brain.prepare_to_iterate()
        self.body.prepare_to_iterate()

        self.brain.iterate()
        self.body.iterate()
        
        if (self.it % int(self.draw_frequency)) == 0 :   
            self.updateDrawnArrays()
            
    def updateDrawnArrays(self):
        pass

if __name__ == '__main__':
    pass
    m = Model()
    init_rvit(m,rvit_file='rvit.kv',window_size=(500,250))
    # skivy.activate()
    # ModelApp().run()
