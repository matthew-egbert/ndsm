from body import Body
from pattern_body import PatternBody
from brain import Brain
from plotting_utils import arena_plot, percent_complete, running_average
import dill as pickle
from rvit.tracker import TrackedValue,TrackingManager
from experiment import Experiment
import numpy as np
from pylab import *

from world import EmptyWorld, World


class PatternExperiment(Experiment):
    def __init__(self,model,name=None) :
        self.model = model
        self.duration = 500000 #float('inf') # 10000

        self.model.training_pattern_length = self.model.TIMESERIES_LENGTH; 
        self.model.world = EmptyWorld(self.model); 
        self.model.body  = PatternBody(self.model, self.model.training_pattern_length, DT=self.model.DT); 
        self.model.brain = Brain(self.model,sm_duration=self.model.training_pattern_length)
        
        if name is None :
            self.name = type(self).__name__ ## gets the class name of the experiment by default
        else :
            self.name = name
    
        self.tracker = TrackingManager(self.name)
        self.EVERY_ITERATION = lambda exp: True
        self.FREQUENTLY      = lambda exp: (m.it % 10) == 0
        self.START           = lambda exp: exp.model.it == 0
        self.END             = lambda exp: exp.model.it == exp.duration-1

        self.tracker.add_pickle_obj('DT',self.model.DT)
        self.tracker.add_pickle_obj('TIMESERIES_LENGTH',self.model.TIMESERIES_LENGTH)                                    
        self.tracker.add_pickle_obj('training_pattern_length',self.model.training_pattern_length)          

        self.tracker.track('time','model.time',should_sample=self.EVERY_ITERATION)
        #self.tracker.track('it','model.it',should_sample=self.EVERY_ITERATION)
        self.tracker.track('x','model.body.x',should_sample=self.EVERY_ITERATION)
        self.tracker.track('y','model.body.y',should_sample=self.EVERY_ITERATION)
        self.tracker.track('α','model.body.α',should_sample=self.EVERY_ITERATION)
        self.tracker.track('prediction_error','model.brain.prediction_error',should_sample=self.EVERY_ITERATION)
        self.tracker.track('sms','model.body.sms',should_sample=self.EVERY_ITERATION)
        #self.tracker.track('deltas','model.brain.deltas[:,:,0]',should_sample=self.EVERY_ITERATION)

    def reset(self) :
        pass


    def iterate(self) :
        percent_complete(self.model.it,self.duration,title='Pattern Experiment')
        self.tracker.iterate(self)
        if self.model.it > self.duration :
            self.end()
        
        if self.model.it > 10000 :
            self.model.body.TRAINING_PHASE = False

    def end(self) :
        print('Experiment completed.')
        self.tracker.save()
        quit()

if __name__ == '__main__':
    path = "PatternExperiment/"

    red = '#8b0000'

    #### POSITION PLOT
    x = np.load(path+'x.npy')
    y = np.load(path+'y.npy')
    time = np.load(path+'time.npy')
    sms = np.load(path+'sms.npy')
    prediction_error = np.load(path+'prediction_error.npy')

    po = pickle.load(open(path+'pickle_objs.pkl','rb'))
    DT = po['DT']
    TIMESERIES_LENGTH = po['TIMESERIES_LENGTH']
    period = TIMESERIES_LENGTH

    figure(figsize=(6,6))
    arena_plot(x,y,-5,5,-5,5)
    tight_layout()
    savefig(path+'position_full.png')
    
    #### POSITION BY TIME SLICES PLOT
    figure(figsize=(12,16))
    R = 8
    C = 6
    N = R*C
    section_length = len(time)//N
    for i in range(N):
        subplot2grid((R,C),(i//C,i%C))
        α = i*section_length;
        ω = (i+1)*section_length;
        title(f'$t\\in${time[α]:.1f}$-${time[ω]:.1f}')
        arena_plot(x[α:ω],y[α:ω],-5,5,-5,5)

    tight_layout()        
    savefig(path+'position_time_slices.png',dpi=300)

    #### TIME SLICED PHASE PLOT
    figure(figsize=(8,8))
    theta = time%(period*DT) / (period*DT) * 2*np.pi        
    r = sms[:,0] * time
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    plot(theta, r,'.',ms=1,label='sensor')
    title('$r=t; θ=sensor$')
    #ax.set_rmax(2)
    gca().set_rticks([])  # Less radial ticks
    gca().set_rlabel_position(-2.5)  # Move radial labels away from plotted line
    #legend()
    #gca().grid(False)
    tight_layout()        
    savefig(path+'phase.png',dpi=300)

    # #### PREDICTION ERROR PLOT
    figure(figsize=(8,5))
    with np.errstate(invalid='ignore'):
        log_prediction_error = np.log(prediction_error,where=prediction_error!=0)

    fill_between(time,log_prediction_error*0-10,log_prediction_error,step='pre',label='log($\\epsilon$)',facecolor=red,alpha=0.7)
    plot(time[:-period],running_average(log_prediction_error,period)[:-period],lw=0.7,color='k',label='running average')
    ylabel('log($\\epsilon$)')
    xlabel('time')
    xlim(0,time[-1])
    ylim(-9,1.8)
    tight_layout()
    savefig(path+'prediction_error.png',dpi=300)

    def cleanplot() :
        xticks([])
        plt.box(False)
        xlim(0,time[-1])

    #### SMS PLOT
    figure(figsize=(8,5))
    
    r,c = 5,1
    subplot2grid((r,c),(0,0))
    fill_between(time,0*sms[:,0],sms[:,0],label='sensor',step='pre',facecolor='y')
    plot(time[:-period],running_average(sms[:,0],period)[:-period],lw=0.7,color='k',label='running average')
    cleanplot()
    ylabel('sensor')
    ticks = arange(0,time[-1],period*DT)
    xticks(ticks)
    gca().set_xticklabels([f' ' for t in ticks])
    gca().xaxis.set_ticks_position('top')
    
    subplot2grid((r,c),(1,0))
    fill_between(time,0*sms[:,0]-1,sms[:,1],label='lm',step='pre')
    plot(time[:-period],running_average(sms[:,1],period)[:-period],lw=0.7,color='k',label='running average')
    cleanplot()
    ylabel('LM')
    
    subplot2grid((r,c),(2,0))
    fill_between(time,0*sms[:,0]-1,sms[:,2],label='rm',step='pre',facecolor='g')
    plot(time[:-period],running_average(sms[:,2],period)[:-period],lw=0.7,color='k',label='running average')
    cleanplot()
    ylabel('RM')
    
    subplot2grid((r,c),(3,0),rowspan=2)
    log_prediction_error = np.log(prediction_error,where=prediction_error!=0)
    fill_between(time,log_prediction_error*0-10,log_prediction_error,step='pre',label='log($\\epsilon$)',facecolor=red,alpha=0.7)
    plot(time[:-period],running_average(log_prediction_error,period)[:-period],lw=0.7,color='k',label='running average')
    ylim(-9,1.8)
    ylabel('log($\\epsilon$)')
    cleanplot()
    ticks = arange(0,time[-1],period*DT)
    xticks(ticks)
    gca().set_xticklabels([f' ' for t in ticks])



    tight_layout()
    savefig(path+'sms.png',dpi=300)


