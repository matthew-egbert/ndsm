from pylab import *

class EmptyWorld :
    def __init__(self):
        self.r = 5.0
        self.walls = []

    def prepare_to_iterate(self) :
        pass
     
    def iterate(self) :
        pass

class BraitenbergWorld(EmptyWorld) :
    def __init__(self, model):
        self.model = model
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
                    ]
        
    def iterate(self):
        super().iterate()

        W = -1.0 * self.r
        E =  1.0 * self.r
        S = -1.0 * self.r
        N =  1.0 * self.r

        M = np.cos(0.015*self.model.it) * 1.5
        r = 0.5
        #print(M)

        c = (1.0+np.cos(0.01*self.model.it)) * 0.3 +0.1

        self.walls = [
                        ((W,S),(W,N)),
                        ((E,S),(E,N)),
                        ((W,N),(E,N)),
                        ((W,S),(E,S)),

                        ((M-r,N),(M,0.6*N)),
                        ((M+r,N),(M,0.6*N)),

                        ((W,c*N),(c*W,N)),
                        ((E,c*N),(c*E,N)),
                        ((W,c*S),(c*W,S)),
                        ((E,c*S),(c*E,S)),
                    ]
    