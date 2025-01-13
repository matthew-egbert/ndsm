import pickle
import matplotlib.patches as patches
import numpy as np
from pylab import *

from experiments.plotters.error_plot import error_plot


def pattern_publication_plots(path) :
    po = pickle.load(open(path+'pickle_objs.pkl','rb'))
    DT = po['DT']
    #Ω = po['Ω']
    #β = po['β']
    training_pattern_length = po['training_pattern_length']
    #training_stop_iteration = po['training_stop_iteration']

    time = np.load(path+'time.npy')
    sms = np.load(path+'sms.npy')
    prediction_error = np.load(path+'prediction_error.npy')

    def sms_slice_plot(index,α,ω,show_sensor=False) :
        data = sms[α:ω,[1,2]]
        xticks([])
        yticks([0.5,1.5])
        gca().set_yticklabels(['RM','LM'],fontsize=8, fontfamily='monospace')
        gca().set_aspect('equal')
        #text(-0.1,1.05 f'$t\in{α*DT:.1f},{ω*DT:.1f}$',fontsize=8,rotation=0,transform=fig.transFigure)
        #ylabel('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[index],fontsize=8,rotation=0)
        fig = plt.gcf()
        if index < 26*2 :
            text(-0.15, 0.8, 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'[index], fontsize=7, rotation=0, transform=gca().transAxes,va='center',ha='center')
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
    error_plot(path)
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

        if index < 26*2 :
            gca().annotate('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'[index], xy=(x, y), xytext=(x,1), 
                           fontsize=8, ha='center', va='bottom', color='k', alpha=1.0,
                           arrowprops=ap)


    tight_layout()
    savefig(path+'pattern_error.png',dpi=300, bbox_inches="tight")