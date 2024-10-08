import torch
from body import Body
from back_and_forth_body import BackAndForthBody
from discval import DiscVal, OneHotter
from pattern_body import PatternBody
from brain import Brain
from plotting_utils import arena_plot, percent_complete, running_average
import dill as pickle
from rvit.tracker import TrackedValue,TrackingManager
from experiment import Experiment
import numpy as np
import networkx as nx
from pylab import *

from world import EmptyWorld, BraitenbergWorld
import matplotlib.patches as patches

class BackAndForthExperiment(Experiment):
    def __init__(self,model,name=None) :
        self.model = model
        self.duration = 51200 #float('inf') # 10000
        self.TRAINING_STOP_ITERATION = 25600

        ## ## BACK AND FORTH
        #self.world : World = EmptyWorld(self); self.body : Body = BackAndForthBody(self,DT=self.DT); self.brain : Brain = Brain(self,sm_duration=64)
        self.model.TIMESERIES_LENGTH = 1024        
        self.model.world = EmptyWorld(self.model); 
        self.model.body = BackAndForthBody(self.model, DT=self.model.DT); 
        self.model.brain = Brain(self.model,Ω=128,β=512)
        
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
        self.tracker.add_pickle_obj('Ω',self.model.brain.Ω)        
        self.tracker.add_pickle_obj('TRAINING_STOP_ITERATION',self.TRAINING_STOP_ITERATION)

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
        self.model.brain.learning_rate_exponent = -3

        title = f'{self.name}'
        if self.model.body.TRAINING_PHASE :
            title += ' [TRAINING]'
        #title += f' lr:{self.model.brain.learning_rate}'
        title += f' err:{self.model.brain.prediction_error:.3f}'
        percent_complete(self.model.it,self.duration,title=title)
        self.tracker.iterate(self)
        
        if self.model.it > self.duration :
            self.end()
        
        if self.model.it > self.TRAINING_STOP_ITERATION :
            self.model.body.TRAINING_PHASE = False
            #self.model.brain.ZERO_LEARNING_RATE = True
            #self.model.brain.learning_rate_exponent = -4

        # if self.model.it % 10000 == 0 :
        #     self.model.brain.image_2d_output()

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
    period = po['Ω']
    TRAINING_STOP_ITERATION = po['TRAINING_STOP_ITERATION']

    # #### POSITION PLOT
    # figure(figsize=(12,5))
    # plot(time,x,label='x')
    # xlabel('time')
    # ylabel('x')
    # tight_layout()
    # savefig(path+'position_full.png')

    # # #### PREDICTION ERROR PLOT
    # figure(figsize=(8,5))
    # with np.errstate(invalid='ignore'):
    #     log_prediction_error = np.log(prediction_error,where=prediction_error!=0)

    # fill_between(time,log_prediction_error*0-10,log_prediction_error,step='pre',label='log($\\epsilon$)',facecolor=red,alpha=0.7)
    # plot(time[:-period],running_average(log_prediction_error,period)[:-period],lw=0.7,color='k',label='running average')
    # ylabel('log($\\epsilon$)')
    # xlabel('time')
    # xlim(0,time[-1])
    # ylim(-4,2.8)
    # tight_layout()
    # savefig(path+'prediction_error.png',dpi=300)

    def cleanplot() :
        xticks([])
        yticks([])
        plt.box(False)
        xlim(-1.1,1.1)
        ylim(-0.1,1.1)

    def sms_slice_plot() :
        #### SENSORIMOTOR SPACE BY TIMESLICE
        print('hi')
        figure(figsize=(12,15))
        R = 8
        C = 8
        N = R*C
        section_length = len(time)//N
        for i in range(N):
            subplot2grid((R,C),(i//C,i%C))
            α = i*section_length;
            ω = (i+1)*section_length;
            #title(f'$t\\in${time[α]:.1f}$-${time[ω]:.1f}')
            print(α,ω)
            # plot(sms[α:ω,1],
            #      sms[α:ω,0],alpha=1.0,color='k')

            for ls_i in range(α,ω-2):
                if α < TRAINING_STOP_ITERATION :
                    color = 'r'
                else :
                    color = 'k'
                # plot(sms[ls_i:ls_i+2,1],
                #      sms[ls_i:ls_i+2,0],alpha=0.1,lw=2.0,color=color)
                μ = 0.03
                plot(sms[ls_i:ls_i+2,1]+np.random.randn(2)*μ,
                    sms[ls_i:ls_i+2,0]+np.random.randn(2)*μ,alpha=0.2,lw=2.0,color=color)
            cleanplot()

        tight_layout()        
        savefig(path+'sms.png',dpi=300)
    
    def timeseries_plot() :
        #### TIMESERIES PLOT
        figure(figsize=(22,12))
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
        fill_between([0,TRAINING_STOP_ITERATION*DT],[-5,-5],[5,5],alpha=0.2,facecolor='k')

        subplot2grid((r,c),(1,0))
        fill_between(time,0*sms[:,0],sms[:,0],label='sensor',step='pre',facecolor='y')
        plot(time[:-period],running_average(sms[:,0],period)[:-period],lw=0.7,color='k',label='running average')
        cleanplot()
        ylabel('sensor')
        
        subplot2grid((r,c),(2,0))
        fill_between(time,0*sms[:,1]-1,sms[:,1],label='lm',step='pre',alpha=0.5)
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
        ticklabels = [f' ' for t in ticks]
        ticklabels[0] = '0'
        ticklabels[-1] = f'{time[-1]:.0f}'
        ticklabels[len(ticklabels)//2] = f'{time[-1]/2:.0f}'
        ticklabels[len(ticklabels)//4] = f'{time[-1]/4:.0f}'
        ticklabels[3*len(ticklabels)//4] = f'{3*time[-1]/4:.0f}'
        xlabel('time')
        gca().set_xticklabels(ticklabels)
        tight_layout()
        savefig(path+'timeseries.png',dpi=300)
        show()

    def error_plot() :
        figure(figsize=(7,2.5))
        ## 512 here is β
        plot(time[512+1:],prediction_error[512+1:],label='log($\\epsilon$)',color='k',lw=0.5)
        minv = prediction_error[512+1:].min()
        maxv = prediction_error[512+1:].max()
        #fill_between(time[],0*prediction_error[512+1:],prediction_error[512+1:],color=red,alpha=0.5)
        rect = patches.Rectangle((0,minv), TRAINING_STOP_ITERATION*DT, maxv+0.4, linewidth=0, edgecolor='w', facecolor='0.666')
        text(2,10**-1.15,'TRAINING PHASE',fontsize=6,ha='left',color='w')
        gca().add_patch(rect)
        yscale('log')
        ylabel('log($\\epsilon$)')
        xlabel('time')
        xlim(0,time[-1])
        tight_layout()
        plt.box(False)
        savefig(path+'osc_error.png',dpi=300)
            
    def position_plot_range(α=0,ω=1024*4) :
        ## 512 here is β
        plot(time[α:ω],x[α:ω],'-',color='k',lw=0.5,label='x')
        minv = prediction_error[512+1:].min()
        maxv = prediction_error[512+1:].max()
        #fill_between(time[],0*prediction_error[512+1:],prediction_error[512+1:],color=red,alpha=0.5)
        #rect = patches.Rectangle((0,minv), TRAINING_STOP_ITERATION*DT, maxv+0.4, linewidth=0, edgecolor='w', facecolor='0.666')
        #text(2,10**-1.15,'TRAINING PHASE',fontsize=6,ha='left',color='w')
        #gca().add_patch(rect)
        #yscale('log')
        ylabel('x',rotation=0)
        #legend(loc='upper right')
        xlabel('time')
        xlim(time[α],time[ω])
        #ylabel(f'$t '+'\in'+f' [{time[α]:.2f},{time[ω]:.2f}]$',fontdict={'fontsize':6})
        #plt.box(False)
    
    def position_plot() :
        figure(figsize=(7,2.5))            
        
        subplot2grid((2,1),(0,0))
        position_plot_range(α=0,ω=1024*4)
        xlabel('')

        # subplot2grid((3,1),(1,0))
        # s = int(350.8/DT)
        # position_plot_range(α=s,ω=s+(1024*4))
        # xlabel('')

        subplot2grid((2,1),(1,0))
        position_plot_range(α=-1024*4-2,ω=-2)        

        tight_layout()
        savefig(path+'osc_position_comparison.png',dpi=300)
        
        # position_plot_range(α=0,ω=1024*4)
        # #savefig(path+'osc_position_start.png',dpi=300)
        
        # position_plot_range(α=-1024*4,ω=-1)
        # savefig(path+'osc_position_end.png',dpi=300)    

    def phase_plot_range(α=0,ω=1024*4,use_position=True,color='k') :
        ## 512 here is β
        for i in range(α,ω-2):
            if use_position :
                xx = x[i:i+2]
            else :
                """otherwise use sensor state"""
                xx = copy(sms[i:i+2,0])
                xx[0] += 0.00005*(i)
                xx[1] += 0.00005*(i+1)
            plot(xx,
                 sms[i:i+2,1],alpha=0.2,lw=0.5,color=color)
        yticks(list(set(sms[:,1])))
        if use_position :
            xlim(-2.5,2.5)
            xlabel('x')
            title(f'$t '+'\\in'+f' [{time[α]:.2f},{time[ω]:.2f}]$')
        else :
            xlim(-0.05,1)
            xlabel('s')
            xticks(list(set(sms[:,0])))
        ylabel('m',rotation=0)
        

    def phase_plot() :
        figure(figsize=(6.0,5.0))            

        subplot2grid((2,2),(0,0))
        phase_plot_range(α=0,ω=1024*4)        
        subplot2grid((2,2),(0,1))
        phase_plot_range(α=-1024*4-2,ω=-2)
        ylabel('')

        subplot2grid((2,2),(1,0))
        phase_plot_range(α=0,ω=1024*4,use_position=False)
        subplot2grid((2,2),(1,1))
        phase_plot_range(α=-1024*4-2,ω=-2,use_position=False)
        ylabel('')

        tight_layout()
        savefig(path+'osc_phase_comparison.png',dpi=300)


    def network_plot(α=0,ω=-1,filename='network.png',label='') :
        if α < 0 :
            α = len(sms)+α
        if ω < 0 :
            ω = len(sms)+ω
        allowed_motor_values = np.linspace(-0.3,0.3,5)
        allowed_sensor_values = np.linspace(0,1,11)
        
        os = DiscVal(allowed_sensor_values, 0, name = "OS")
        om = DiscVal(allowed_motor_values, 0, name = "OM")
        oh = OneHotter([os,om])
        
        ## num sensorimotor states
        N = len(oh.onehot)
        
        transition_counts = np.zeros((N,N))
        
        label_mapping = dict()
        pos = dict()
        
        for i in range(α,ω-2):
            s1,m1 = sms[i]
            s2,m2 = sms[i+1]
            oh.values = [s1,m1]
            i1 = np.argmax(oh.onehot)
            label_mapping[i1] = f'{s1:.2f},{m1:.2f}' # f'{s1*10:.0f},{m1*10:.0f}'
            pos[label_mapping[i1]] = (m1,s1)
            

            oh.values = [s2,m2]
            i2 = np.argmax(oh.onehot)

            transition_counts[i2,i1] += 1
        
        transition_counts = log(transition_counts+1)
        figure(figsize=(6,4.5))
        imshow(transition_counts,cmap='viridis')
        colorbar()
        tight_layout()
        savefig(path+'transition_count_matrix.png',dpi=300)
        close()
        
        figure(figsize=(4.5,4.5))
        xlim(-0.35,0.35)
        ylim(-0.05,0.98)
        x0, y0 = gca().transAxes.transform((-0.35, -0.05)) # lower left in pixels
        x1, y1 = gca().transAxes.transform((0.35, 0.98)) # upper right in pixes
        dy = x1 - x0
        dx = y1 - y0
        maxd = max(dx, dy)
        width = .02 * maxd / dx
        height = .02 * maxd / dy
        
        G = nx.DiGraph(transition_counts)
        G.remove_nodes_from(list(nx.isolates(G)))        
        G = nx.relabel_nodes(G, label_mapping)        
        # for node in G.nodes() :
        #     G.nodes[node]['pos'] = (float(node.split(',')[0]), float(node.split(',')[1]))
        
        #nx.draw(G, pos, with_labels=True, node_size=200, node_color='skyblue', font_size=12, font_color='black', font_weight='bold', edge_color='gray', width=0.5, arrowsize=5)
        #for node in G.nodes() :
            #gca().add_artist(patches.Ellipse((pos[node][0], pos[node][1]), width, height))
            #text = gca().text(pos[node][0],pos[node][1],node,ha='center',va='center',fontsize=10)
            #gca().add_artist(text)
        for edge in G.edges():
            source, target = edge
            rad = 0.2
            arrowprops=dict(lw=G.edges[(source,target)]['weight'],
                            arrowstyle="->",
                            color='black',
                            connectionstyle=f"arc3,rad={rad}",
                            linestyle= '-',
                            shrinkA=2.0,
                            shrinkB=2.0,
                            alpha=1.0)
            gca().annotate("",
                        xy=pos[source],
                        xytext=pos[target],
                        arrowprops=arrowprops
                    )
        
        # xticks([])
        # yticks([])
        xlabel('m',fontsize=12)
        ylabel('s',fontsize=12,rotation=0)
        plt.text(0.0, 1.0, label, fontsize=28, ha='center', fontweight='bold', transform=plt.gca().transAxes)
        plt.text(0.5,1.0,f'$t \\in[{time[α]:.2f},{time[ω]:.2f})$',fontsize=14,ha='center', transform=plt.gca().transAxes)
        plt.box(False)
        tight_layout()
        savefig(path+filename,dpi=300)


    def network_plots() :
        p = int(3.11//DT) ## oscilalation length in iterations (approximated visually)
        n = 36 ## number of oscillations
        #network_plot(0,-1,'network_full.png')#-1000,-1)
        network_plot(0,p*n,'network_start.png',label='A')#-1000,-1)
        network_plot(-p*n,-1,'network_end.png',label='B')#-1000,-1)
        
            

    #error_plot()
    #show()
    position_plot()
    #network_plots()

