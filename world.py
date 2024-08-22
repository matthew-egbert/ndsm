from pylab import *

class World() :
    def __init__(self, *args, **kwargs):
        self.r = 5.0
        c = 0.8

        W = -1.0 * self.r
        E =  1.0 * self.r
        S = -1.0 * self.r
        N =  1.0 * self.r

        self.walls = [
                      ((W,S),(W,N)),
                      ((E,S),(E,N)),
                      ((W,N),(E,N)),
                      ((W,S),(E,S)),

                      ((0.4*W,N),(0.0,0.6*N)),
                      ((0.4*E,N),(0.0,0.6*N)),

                      ((W,c*N),(c*W,N)),
                      ((E,c*N),(c*E,N)),
                      ((W,c*S),(c*W,S)),
                      ((E,c*S),(c*E,S)),


                    #   ((-0.5,0.15),(0.3,0.15)),
                    #   ((0.3,0.05),(0.3,0.15)),
                    #   ((-0.5,0.05),(0.3,0.05)),
                    #   ((-0.5,0.05),(-0.5,0.15)),
                     ]
        
        # k = 2.0
        # self.walls = [tuple(k * np.array(wall)) for wall in self.walls]