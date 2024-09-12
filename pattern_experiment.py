from body import Body
from pattern_body import PatternBody
from brain import Brain
from plotting_utils import arena_plot, percent_complete, running_average
import dill as pickle
from rvit.tracker import TrackedValue,TrackingManager
from experiment import Experiment
import numpy as np
from pylab import *

from world import EmptyWorld, BraitenbergWorld


class PatternExperiment(Experiment):
    def __init__(self,model,name=None) :
        self.model = model
        self.TRAINING_STOP_ITERATION =  51200 /8
        self.duration                = 102400 /4 # 100000

        self.model.TIMESERIES_LENGTH = 1024
        self.training_pattern_length = 64

        self.model.world = EmptyWorld(self.model); 
        self.model.body  = PatternBody(self.model, self.training_pattern_length, DT=self.model.DT); 
        self.model.brain = Brain(self.model,Ω=self.training_pattern_length,β=512)

        
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
        self.tracker.add_pickle_obj('training_pattern_length',self.training_pattern_length)          
        self.tracker.add_pickle_obj('TRAINING_STOP_ITERATION',self.TRAINING_STOP_ITERATION)

        self.tracker.track('time','model.time',should_sample=self.EVERY_ITERATION)
        self.tracker.track('it','model.it',should_sample=self.EVERY_ITERATION)
        self.tracker.track('x','model.body.x',should_sample=self.EVERY_ITERATION)
        self.tracker.track('y','model.body.y',should_sample=self.EVERY_ITERATION)
        self.tracker.track('α','model.body.α',should_sample=self.EVERY_ITERATION)
        self.tracker.track('prediction_error','model.brain.prediction_error',should_sample=self.EVERY_ITERATION)
        self.tracker.track('sms','model.body.sms',should_sample=self.EVERY_ITERATION)

    def reset(self) :
        pass

    def iterate(self) :
        color = 'c'
        if self.model.body.TRAINING_PHASE :
            color = 'r'
        pe = self.model.brain.prediction_error
        if pe != 0 :
            pe = log(pe)
        percent_complete(self.model.it,self.duration,title=f'Pattern Exp. error exponent={pe:.3f}',color=color)
        self.tracker.iterate(self)
        if self.model.it > self.duration :
            self.end()
        
        if self.model.it > self.TRAINING_STOP_ITERATION :
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

    period = po['training_pattern_length']
    TRAINING_STOP_ITERATION = po['TRAINING_STOP_ITERATION']

    # figure(figsize=(6,6))
    # #arena_plot(x[0:TRAINING_STOP_ITERATION],y[0:TRAINING_STOP_ITERATION],-5,5,-5,5,color='r')
    # #arena_plot(x[TRAINING_STOP_ITERATION:],y[TRAINING_STOP_ITERATION:],-5,5,-5,5,color='k')
    # α,ω = 0,len(time)
    
    # for σ in range(α,ω):
    #         percent_complete(σ,len(time),title='Plotting Position',color='y',bar_width=30)
    #         if σ < TRAINING_STOP_ITERATION :
    #             color = 'c'
    #         else :
    #             color = 'k'
    #         arena_plot(x[σ:σ+2],y[σ:σ+2],-5,5,-5,5,alpha=0.5,color=color)
    #         #arena_plot(x[σ:σ+step],y[σ:σ+step],alpha=0.1,color=color)
    # xlim(-5,5)
    # ylim(-5,5)

    # tight_layout()
    # savefig(path+'position_full.png',dpi=300)
    
    #### POSITION BY TIME SLICES PLOT
    figure(figsize=(12,16))
    R = 10
    C = 8
    N = R*C
    section_length = len(time)//N
    axs =[]    
    for i in range(N):
        axs.append( subplot2grid((R,C),(i//C,i%C)) )
    for i in range(N):
        plt.sca(axs[i])
        α = i*section_length;
        ω = (i+1)*section_length;
        #title(f'$t\\in${time[α]:.1f}$-${time[ω]:.1f}')
        if α < TRAINING_STOP_ITERATION :
            color = 'r'
        else :
            color = 'k'
        arena_plot(x[α:ω],y[α:ω],-5,5,-5,5,color=color)
        xticks([])
        yticks([])

    tight_layout()        
    savefig(path+'position_time_slices.png',dpi=300)

    # #### PHASE PLOT
    # figure(figsize=(8,8))
    # theta = time%(period*DT) / (period*DT) * 2*np.pi        
    # r = sms[:,0] * time
    # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # plot(theta, r,'.',ms=1,label='sensor')
    # title('$r=t; θ=sensor$')
    # #ax.set_rmax(2)
    # gca().set_rticks([])  # Less radial ticks
    # gca().set_rlabel_position(-2.5)  # Move radial labels away from plotted line
    # #legend()
    # #gca().grid(False)
    # tight_layout()        
    # savefig(path+'phase.png',dpi=300)

    def cleanplot() :
        xticks([])
        plt.box(False)
        xlim(0,time[-1])

    #### SMS PLOT
    figure(figsize=(8,5))
    
    r,c = 4,1
    subplot2grid((r,c),(0,0))   
    fill_between(time,0*sms[:,0]-1,sms[:,-2],label='lm',step='pre')
    plot(time[:-period],running_average(sms[:,-2],period)[:-period],lw=0.7,color='k',label='running average')
    cleanplot()
    ylabel('LM')
    
    subplot2grid((r,c),(1,0))   
    fill_between(time,0*sms[:,0]-1,sms[:,-1],label='rm',step='pre',facecolor='g')
    plot(time[:-period],running_average(sms[:,-1],period)[:-period],lw=0.7,color='k',label='running average')
    cleanplot()
    ylabel('RM')

    # subplot2grid((r,c),(2,0))
    # fill_between(time,0*sms[:,0],sms[:,0],label='sensor',step='pre',facecolor='y')
    # plot(time[:-period],running_average(sms[:,0],period)[:-period],lw=0.7,color='k',label='running average')
    # cleanplot()
    # ylabel('sensor')
    # ticks = arange(0,time[-1],period*DT)
    # xticks(ticks)
    # gca().set_xticklabels([f' ' for t in ticks])
    # gca().xaxis.set_ticks_position('top')


    subplot2grid((r,c),(2,0),rowspan=2)
    log_prediction_error = np.log(prediction_error,where=prediction_error!=0)
    fill_between(time,log_prediction_error*0-10,log_prediction_error,step='pre',label='log($\\epsilon$)',facecolor=red,alpha=0.7)
    plot(time[:-period],running_average(log_prediction_error,period)[:-period],lw=0.7,color='k',label='running average')
    ylim(-12,1.8)
    ylabel('log($\\epsilon$)')
    cleanplot()
    ticks = arange(0,time[-1],period*DT)
    xticks(ticks)
    ticklabels = [f' ' for t in ticks]
    ticklabels[0] = '0'
    ticklabels[-1] = f'{time[-1]:.0f}'
    ticklabels[len(ticklabels)//2] = f'{time[-1]/2:.0f}'
    ticklabels[len(ticklabels)//4] = f'{time[-1]/4:.0f}'
    ticklabels[3*len(ticklabels)//4] = f'{3*time[-1]/4:.0f}'
    gca().set_xticklabels(ticklabels)
    xlabel('time')
    tight_layout()
    savefig(path+'sms.png',dpi=300)


