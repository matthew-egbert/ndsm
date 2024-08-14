from pylab import *
from scipy.spatial.distance import cdist 
from SweepIntersectorLib.SweepIntersector import SweepIntersector
from smcodec import SMCodec


LEFT = 0
RIGHT = 1

class Body(object) :
    def __init__(self, model, DT = 0.01, radius = 0.1, sensor_length = 1.0, β=np.pi/3, γ=np.pi/4) :
        """ β : gap between L/R sensor clusters
            γ : radial width of each sensor cluster 
        """

        self.model : Model = model
        self.x = 0.0
        self.y = 0.0
        self.α = 0.0
        self.N_ALLOWED_SENSOR_VALUES = 2
        self.N_ALLOWED_MOTOR_VALUES = 5
        self.ALLOWED_SENSOR_VALUES = np.linspace(0,1,self.N_ALLOWED_SENSOR_VALUES)
        self.ALLOWED_MOTOR_VALUES = np.linspace(-1,1,self.N_ALLOWED_MOTOR_VALUES)
        
        self.set_motors(2+0,2+0)
        self.r = radius
        self.DT = DT        
        self.sensor_length = sensor_length

        self.n_sensors = 2; assert((self.n_sensors % 2) == 0)
        self.n_motors = 2

        sensors_per_side = self.n_sensors // 2
        self.sensor_excitations = np.zeros(self.n_sensors)
        self.l_sensors_βs = np.linspace(-β-γ/2,-β+γ/2,sensors_per_side)
        self.r_sensors_βs = np.linspace(β-γ/2,β+γ/2,sensors_per_side)
        self.sensors_βs = concatenate([self.l_sensors_βs,self.r_sensors_βs])

        self.SMS = np.random.rand(self.n_sensors+self.n_motors)

        self.h_length = self.model.TIMESERIES_LENGTH
        self.s_h = np.zeros((self.n_sensors,self.h_length)) ## sensor_history ## TODO: can I have s_h and m_h be views of sms_h?
        self.m_h = np.zeros((2,self.h_length))
        self.sms_h = np.zeros((self.n_sensors+2,self.h_length))

        self.smcodec = SMCodec([self.ALLOWED_SENSOR_VALUES,self.ALLOWED_SENSOR_VALUES,self.ALLOWED_MOTOR_VALUES,self.ALLOWED_MOTOR_VALUES])
        #self.test_motor_mappings()

    def set_motors(self,lmi,rmi):
        assert(lmi < self.N_ALLOWED_MOTOR_VALUES)
        assert(rmi < self.N_ALLOWED_MOTOR_VALUES)
        assert(lmi >= 0)
        assert(rmi >= 0)

        self.LMV = lmi
        self.RMV = rmi
        self.lm = self.ALLOWED_MOTOR_VALUES[self.LMV]
        self.rm = self.ALLOWED_MOTOR_VALUES[self.RMV]

    def test_motor_mappings(self) :        
        for i in range(self.N_ALLOWED_SENSOR_VALUES) :
            for j in range(self.N_ALLOWED_SENSOR_VALUES) :
                for k in range(self.N_ALLOWED_MOTOR_VALUES) :                
                    for l in range(self.N_ALLOWED_MOTOR_VALUES) :                        
                        ls = self.ALLOWED_SENSOR_VALUES[i]
                        rs = self.ALLOWED_SENSOR_VALUES[j]                
                        lm = self.ALLOWED_MOTOR_VALUES[k]
                        rm = self.ALLOWED_MOTOR_VALUES[l]
                        oneshot = self.smcodec.to_onehot( (ls,rs,lm,rm) )
                        ls2,rs2,lm2, rm2 = self.smcodec.from_onehot(oneshot)
                        print(f'{ls} {rs} {lm} {rm} \t=> {argmax(oneshot)} \t=> {ls2} {rs2} {lm2} {rm2}')

                        assert(ls == ls2)
                        assert(rs == rs2)
                        assert(lm == lm2)
                        assert(rm == rm2)
        quit()
                
    def update_sensors(self):
        ## sensor proximal coords
        self.spxs = self.x + np.cos(self.α+self.sensors_βs) * self.r
        self.spys = self.y + np.sin(self.α+self.sensors_βs) * self.r
        ## sensor distal coords
        self.sdxs = self.x + np.cos(self.α+self.sensors_βs) * (self.r+self.sensor_length) 
        self.sdys = self.y + np.sin(self.α+self.sensors_βs) * (self.r+self.sensor_length)

        self.sensor_line_segments = [((xp,yp),(xd,yd)) for xp,yp,xd,yd in zip(self.spxs,self.spys,self.sdxs,self.sdys)]

        ## line segments to check for intersections
        line_segs = self.model.world.walls + self.sensor_line_segments
        # compute intersections        
        self.isector = SweepIntersector()   
        intersections = self.isector.findIntersections(line_segs)

        for sensor_index, sensor_seg in enumerate(self.sensor_line_segments) :
            intersection_locations = np.array([loc for loc in intersections[sensor_seg][1:-1]]).reshape(-1,2)
            distances = cdist(np.array(sensor_seg[0]).reshape(-1,2),intersection_locations)[0]
            #print(f'{sensor_index}: {intersection_locations} ::: {distances}')
            if(len(distances) > 0 ):
                closest = argmin(distances)                        
                self.sensor_excitations[sensor_index] = 1.0-(distances[closest] / self.sensor_length)
            else :
                self.sensor_excitations[sensor_index] = 0.0

        self.sensor_excitations[self.sensor_excitations > 0.5] = 1.0
        self.sensor_excitations[self.sensor_excitations <= 0.5] = 0.0
        self.SMS[:self.n_sensors] = self.sensor_excitations


    def debug_plot(self) :
        self.x = 0.5
        self.y = 0.5
        self.α = np.random.rand()*2.0*np.pi

        self.update_sensors()
        
        ### plot body
        figure(figsize=(12,5))
        subplot2grid((1,2),(0,0))
        body = plt.Circle((self.x, self.y), self.r, fc='#cccccc',ec='k')
        gca().add_artist(body)
        heading = plt.Line2D([self.x,self.x+np.cos(self.α)*self.r],
                             [self.y,self.y+np.sin(self.α)*self.r],color='k')
        gca().add_artist(heading)

        ### plot sensor lines
        for sensor_i,((x,y),(x2,y2)) in enumerate(self.sensor_line_segments) :
            excitation = self.sensor_excitations[sensor_i]
            whisker = plt.Line2D([x,x2],[y,y2],color='k',alpha=excitation)
            gca().add_artist(whisker)
        
        ### plot sensable walls
        for ((x,y),(x2,y2)) in self.model.world.walls :
            wall = plt.Line2D([x,x2],[y,y2])
            gca().add_artist(wall)

        xlim(-0.2,1.2)
        ylim(-0.2,1.2)
        gca().set_aspect('equal') 

        ## plot values of sensors
        subplot2grid((1,2),(0,1))
        step(range(len(self.sensor_excitations)),self.sensor_excitations)
        xlabel('sensor index')       
        ylabel('excitation')       

    def randomizePosition(self) :
        self.x = np.random.rand()
        self.y = np.random.rand()
        self.α = np.random.rand()*2.0*np.pi

    def prepare_to_iterate(self) :
        pass
        #self.lm = self.model.brain.get_output(0)
        #self.rm = self.model.brain.get_output(1)

    def iterate(self) :
        self.update_sensors()

        ### braitenberg vehicle
        if self.sensor_excitations[RIGHT] > 0.5 :
            self.set_motors(2+1,2-1)
        else :
            self.set_motors(2+1,2+1)

        #self.state_history.append([self.x,self.y,self.α])
        self.s_h[:,self.model.it%self.h_length] = self.sensor_excitations
        self.m_h[:,self.model.it%self.h_length] = [self.lm,self.rm]
        self.sms_h[:,self.model.it%self.h_length] = self.sensor_excitations[0],self.sensor_excitations[1],self.lm,self.rm
        self.dx = 0.0
        self.dy = 0.0

        r = 1
        self.dx = cos(self.α)*(self.lm+self.rm)*r
        self.dy = sin(self.α)*(self.lm+self.rm)*r
        self.da = 2.0*(self.rm-self.lm)*r*5.0

        self.x += self.DT * self.dx
        self.y += self.DT * self.dy
        self.α += self.DT * self.da

        ω = 5.0 # world radius
        if self.x > ω : 
            self.x -= 2.0*ω
        if self.y > ω : 
            self.y -= 2.0*ω

        if self.x < -ω : 
            self.x += 2.0*ω
        if self.y < -ω : 
            self.y += 2.0*ω

if __name__ == '__main__' :
    from main import Model
    r = Body(Model())
    r.debug_plot()
    show()