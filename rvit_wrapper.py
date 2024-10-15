from time import sleep
import os
os.environ["KIVY_NO_ARGS"] = "1"
from kivy.config import Config
from kivy import platform
Config.set('kivy', 'log_level', 'debug')  # Set the log level to 'debug'

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


from kivy.clock import Clock
from rvit.core import init_rvit # type: ignore


def attach_rvit(m):            
    def iterate(arg) :
        sleep(m.sleep_amount)

        if m.sleep_amount < 0.5 : ## PAUSE if at max delay
            m.iterate()

    Clock.schedule_interval(iterate, 0.0)
    init_rvit(m,rvit_file='rvit.kv',window_size=(500,250))    