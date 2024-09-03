import torch
from body import Body
from back_and_forth_body import BackAndForthBody
from pattern_body import PatternBody
from brain import Brain
from plotting_utils import arena_plot, percent_complete, running_average
import dill as pickle
from rvit.tracker import TrackedValue,TrackingManager
from experiment import Experiment
import numpy as np
from pylab import *

from world import EmptyWorld, World


class BackAndForthExperiment(Experiment):
    def __init__(self,model,name=None) :
        self.model = model
        self.duration = 30000 #float('inf') # 10000


        ## ## BACK AND FORTH
        #self.world : World = EmptyWorld(self); self.body : Body = BackAndForthBody(self,DT=self.DT); self.brain : Brain = Brain(self,sm_duration=64)
        self.model.TIMESERIES_LENGTH = 128
        self.model.training_pattern_length = self.model.TIMESERIES_LENGTH; 
        self.model.world = EmptyWorld(self.model); 
        self.model.body = BackAndForthBody(self.model, DT=self.model.DT); 
        self.model.brain = Brain(self.model,sm_duration=48)
        
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

    # def poincare(self) :
    #     brain = self.model.brain
    #     body = self.model.body
    #     ss = body.sensors[0].allowed_values
    #     ms = body.motors[0].allowed_values
    #     grid = meshgrid(ss,ms)

    #     def sample_brain_output(x) :
    #         probs = brain.n_model(torch.tensor(x).float().to(brain.device)).detach().cpu().numpy()
    #         onehot_i = argmax(probs)
    #         onehot = np.zeros(len(probs))
    #         onehot[onehot_i] = 1
    #         body.onehotter.onehot = onehot
    #         output_values = self.body.onehotter.values        
    #         print(output_values)
    #         quit()

    #     nn_out = np.apply_along_axis(sample_brain_output,0,grid)

    #     quit()

    def iterate(self) :
        self.model.brain.learning_rate_exponent = -3

        title = f'{self.name}'
        if self.model.body.TRAINING_PHASE :
            title += ' [TRAINING]'
        title += f' lr:{self.model.brain.learning_rate}'
        percent_complete(self.model.it,self.duration,title=title)
        self.tracker.iterate(self)
        
        if self.model.it > self.duration :
            self.end()
        
        if self.model.it > 10000 :
            self.model.body.TRAINING_PHASE = False
            #self.model.brain.ZERO_LEARNING_RATE = True
            #self.model.brain.learning_rate_exponent = -4

    def end(self) :
        print('Experiment completed.')
        self.tracker.save()
        quit()

if __name__ == '__main__':
    path = "BackAndForthExperiment/"
    red = '#8b0000'
    x = np.load(path+'x.npy')
    time = np.load(path+'time.npy')
    sms = np.load(path+'sms.npy')
    prediction_error = np.load(path+'prediction_error.npy')

    po = pickle.load(open(path+'pickle_objs.pkl','rb'))
    DT = po['DT']
    TIMESERIES_LENGTH = po['TIMESERIES_LENGTH']
    period = TIMESERIES_LENGTH

    # #### POSITION PLOT
    # figure(figsize=(12,5))
    # plot(time,x,label='x')
    # xlabel('time')
    # ylabel('x')
    # tight_layout()
    # savefig(path+'position_full.png')

    # #### PREDICTION ERROR PLOT
    figure(figsize=(8,5))
    with np.errstate(invalid='ignore'):
        log_prediction_error = np.log(prediction_error,where=prediction_error!=0)

    fill_between(time,log_prediction_error*0-10,log_prediction_error,step='pre',label='log($\\epsilon$)',facecolor=red,alpha=0.7)
    plot(time[:-period],running_average(log_prediction_error,period)[:-period],lw=0.7,color='k',label='running average')
    ylabel('log($\\epsilon$)')
    xlabel('time')
    xlim(0,time[-1])
    ylim(-4,2.8)
    tight_layout()
    savefig(path+'prediction_error.png',dpi=300)

    def cleanplot() :
        xticks([])
        yticks([])
        plt.box(False)
        xlim(-1.1,1.1)
        ylim(-0.1,1.1)

    # #### POSITION BY TIME SLICES PLOT
    # figure(figsize=(12,15))
    # R = 8
    # C = 8
    # N = R*C
    # section_length = len(time)//N
    # for i in range(N):
    #     subplot2grid((R,C),(i//C,i%C))
    #     α = i*section_length;
    #     ω = (i+1)*section_length;
    #     #title(f'$t\\in${time[α]:.1f}$-${time[ω]:.1f}')
    #     print(α,ω)
    #     # plot(sms[α:ω,1],
    #     #      sms[α:ω,0],alpha=1.0,color='k')

    #     for ls_i in range(α,ω-2):
    #         plot(sms[ls_i:ls_i+2,1],
    #              sms[ls_i:ls_i+2,0],alpha=0.1,color='k')
    #     cleanplot()

    # tight_layout()        
    # savefig(path+'sms.png',dpi=300)
    
    #### SMS PLOT
    figure(figsize=(8,12))
    def cleanplot() :
        xticks([])
        #yticks([])
        plt.box(False)
        # xlim(-1,1)
        # ylim(0,1)
    
    r,c = 5,1
    subplot2grid((r,c),(0,0))
    plot(time,x,lw=0.7,color='k')
    cleanplot()
    ylabel('x')
    ticks = arange(0,time[-1],period*DT)
    xticks(ticks)
    gca().set_xticklabels([f' ' for t in ticks])
    gca().xaxis.set_ticks_position('top')


    subplot2grid((r,c),(1,0))
    fill_between(time,0*sms[:,0],sms[:,0],label='sensor',step='pre',facecolor='y')
    plot(time[:-period],running_average(sms[:,0],period)[:-period],lw=0.7,color='k',label='running average')
    cleanplot()
    ylabel('sensor')
    
    subplot2grid((r,c),(2,0))
    fill_between(time,0*sms[:,1]-1,sms[:,1],label='lm',step='pre')
    plot(time[:-period],running_average(sms[:,1],period)[:-period],lw=0.7,color='k',label='running average')
    cleanplot()
    ylim(sms[:,1].min(),sms[:,1].max())
    ylabel('M')
        
    subplot2grid((r,c),(3,0),rowspan=2)
    log_prediction_error = np.log(prediction_error,where=prediction_error!=0)
    fill_between(time,log_prediction_error*0-10,log_prediction_error,step='pre',label='log($\\epsilon$)',facecolor=red,alpha=0.7)
    plot(time[:-period],running_average(log_prediction_error,period)[:-period],lw=0.7,color='k',label='running average')
    ylim(-4,1.8)
    ylabel('log($\\epsilon$)')
    cleanplot()
    ticks = arange(0,time[-1],period*DT)
    xticks(ticks)
    gca().set_xticklabels([f' ' for t in ticks])
    savefig(path+'timeseries.png',dpi=300)

