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
import matplotlib.patches as patches


class PatternExperiment(Experiment):
    def __init__(self,model,name=None) :
        self.model = model
        self.TRAINING_STOP_ITERATION =  51200 /8
        self.duration                = 102400 /4 # 100000

        self.model.TIMESERIES_LENGTH = 1024
        self.training_pattern_length = 64
        self.β = 512

        self.model.world = EmptyWorld(self.model);
        self.model.body  = PatternBody(self.model, self.training_pattern_length, DT=self.model.DT);
        self.model.brain = Brain(self.model,Ω=self.training_pattern_length,β=self.β)


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
        self.tracker.add_pickle_obj('β',self.β)

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
    x = np.load(path+'x.npy')
    y = np.load(path+'y.npy')
    time = np.load(path+'time.npy')
    sms = np.load(path+'sms.npy')
    prediction_error = np.load(path+'prediction_error.npy')

    po = pickle.load(open(path+'pickle_objs.pkl','rb'))
    DT = po['DT']
    training_pattern_length = po['training_pattern_length']
    TRAINING_STOP_ITERATION = po['TRAINING_STOP_ITERATION']

    #### POSITION PLOT
    def position_plot() :
        #    figure(figsize=(6,6))
        #arena_plot(x[0:TRAINING_STOP_ITERATION],y[0:TRAINING_STOP_ITERATION],-5,5,-5,5,color='r')
        #arena_plot(x[TRAINING_STOP_ITERATION:],y[TRAINING_STOP_ITERATION:],-5,5,-5,5,color='k')
        α,ω = 0,len(time)

        for σ in range(α,ω):
                percent_complete(σ,len(time),title='Plotting Position',color='y',bar_width=30)
                if σ < TRAINING_STOP_ITERATION :
                    color = 'c'
                else :
                    color = 'k'
                arena_plot(x[σ:σ+2],y[σ:σ+2],-5,5,-5,5,alpha=0.5,color=color)
                #arena_plot(x[σ:σ+step],y[σ:σ+step],alpha=0.1,color=color)

        tight_layout()
        savefig(path+'position_full.png',dpi=300)

    #### POSITION BY TIME SLICES PLOT
    def position_time_slices_plot() :
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

    #### PHASE PLOT
    def phase_plot() :
        figure(figsize=(8,8))
        theta = time%(training_pattern_length*DT) / (training_pattern_length*DT) * 2*np.pi
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

    def cleanplot() :
        xticks([])
        plt.box(False)
        xlim(0,time[-1])

    def timeseries_plot() :
        #### TIMESERIES PLOTS
        figure(figsize=(8,5))

        r,c = 4,1
        subplot2grid((r,c),(0,0))
        fill_between(time,0*sms[:,0]-1,sms[:,-2],label='lm',step='pre')
        plot(time[:-training_pattern_length],running_average(sms[:,-2],training_pattern_length)[:-training_pattern_length],lw=0.7,color='k',label='running average')
        cleanplot()
        ylabel('LM')

        subplot2grid((r,c),(1,0))
        fill_between(time,0*sms[:,0]-1,sms[:,-1],label='rm',step='pre',facecolor='g')
        plot(time[:-training_pattern_length],running_average(sms[:,-1],training_pattern_length)[:-training_pattern_length],lw=0.7,color='k',label='running average')
        cleanplot()
        ylabel('RM')

        # subplot2grid((r,c),(2,0))
        # fill_between(time,0*sms[:,0],sms[:,0],label='sensor',step='pre',facecolor='y')
        # plot(time[:-training_pattern_length],running_average(sms[:,0],training_pattern_length)[:-training_pattern_length],lw=0.7,color='k',label='running average')
        # cleanplot()
        # ylabel('sensor')
        # ticks = arange(0,time[-1],training_pattern_length*DT)
        # xticks(ticks)
        # gca().set_xticklabels([f' ' for t in ticks])
        # gca().xaxis.set_ticks_position('top')

    def error_plot() :
        ## 512 here is β
        plot(time[512+1:],prediction_error[512+1:],label='log($\\epsilon$)',color='k',lw=1.0)
        minv = prediction_error[512+1:].min()
        maxv = prediction_error[512+1:].max()
        #fill_between(time[],0*prediction_error[512+1:],prediction_error[512+1:],color=red,alpha=0.5)
        rect = patches.Rectangle((0,minv), TRAINING_STOP_ITERATION*DT, maxv+0.4, linewidth=0, edgecolor='w', facecolor='0.666')
        text(2,10**-3.2,'TRAINING PHASE',fontsize=6,ha='left',color='w')
        gca().add_patch(rect)
        yscale('log')
        ylabel('log($\\epsilon$)')
        xlabel('time')
        xlim(0,time[-1])
        tight_layout()
        savefig(path+'timeseries.png',dpi=300)

    ## Publication plot
    def publication_plot() :
        def sms_slice_plot(index,α,ω,show_sensor=False) :
            data = sms[α:ω,[1,2]]
            xticks([])
            yticks([0.5,1.5])
            gca().set_yticklabels(['RM','LM'],fontsize=8, fontfamily='monospace')
            gca().set_aspect('equal')
            #text(-0.1,1.05 f'$t\in{α*DT:.1f},{ω*DT:.1f}$',fontsize=8,rotation=0,transform=fig.transFigure)
            #ylabel('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[index],fontsize=8,rotation=0)
            fig = plt.gcf()
            text(-0.15, 0.8, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[index], fontsize=7, rotation=0, transform=gca().transAxes,va='center',ha='center')
            text(-0.15, 0.2, f'{α*DT:.2f}-{ω*DT:.2f}', fontsize=7, rotation=0, transform=gca().transAxes,va='center',ha='center')
            plt.box(False)
            ylim(-0.05,2.0)
            xlim(α,ω)

            for it in range(data.shape[0]):
                for sm_i in range(data.shape[1]):
                    edgecolor='w'
                    facecolor = {
                        -2: '0.8',
                        2 : '0.0',
                    }[data[it,sm_i]]

                    if data[it,sm_i] != sms[:,1:3][it%training_pattern_length,sm_i] :
                        #edgecolor = 'r'
                        facecolor = 'r'
                        #rect = patches.Rectangle(((α+it),1-sm_i), 1, 1, linewidth=1, edgecolor=edgecolor, facecolor=facecolor)
                        rect = patches.Circle(((α+it)+0.5,1-sm_i+0.5), 0.5, linewidth=1, edgecolor=edgecolor, facecolor=facecolor)
                    else :
                        rect = patches.Rectangle(((α+it),1-sm_i), 1, 1, linewidth=1, edgecolor=edgecolor, facecolor=facecolor)
                    gca().add_patch(rect)
                    

        
        #ts = [0, 80.8, 88.5, 127.5, 140.0, 250.0]
        ts = []
        for i in range(0,len(time)-training_pattern_length,training_pattern_length) :
            for j in range(0,training_pattern_length) :
                arr1 = sms[i+j,:]
                arr2 = sms[j,:]
                if not np.all(arr1 == arr2) :
                    ts.append(i*DT)
                    break
        
        ts = [0,]+ts

        def t_to_aw(t) :
            s = int(training_pattern_length)
            a = int((t/DT)//s)*s
            w = a + s
            return a,w

        aws = [t_to_aw(t) for t in ts]        
        #aws = set(aws)

        fig_width = 7
        fig = figure(figsize=(fig_width,7))
        rows = len(aws)
        for index,aw in enumerate(aws) :
            subplot2grid((rows,1),(index,0))
            sms_slice_plot(index,aw[0],aw[1])
        tight_layout()        
        savefig(path+'pattern_details.png',dpi=300, bbox_inches="tight")

        figure(figsize=(fig_width,2.5))
        error_plot()
        plt.box(False)
        for index,(a,w) in enumerate(aws) :
            x = (a*DT)#/time[-1]
            y = prediction_error[a]

            if index == 0:
                x = 2
                y = 0.0005;
                ap = None
            else :
                ap = dict(arrowstyle='-', color='0.0', lw=0.8, shrinkB=0)

            gca().annotate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[index], xy=(x, y), xytext=(x,1), fontsize=8, ha='center', va='bottom', color='k', alpha=1.0,
                            arrowprops=ap)
       

        tight_layout()
        # fig.subplots_adjust(hspace=0.0)
        savefig(path+'pattern_error.png',dpi=300, bbox_inches="tight")

    # position_plot()
    # position_time_slices_plot()
    # phase_plot()
    # timeseries_plot()
    # error_plot()
    publication_plot()
    #show()



