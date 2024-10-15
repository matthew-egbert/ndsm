from pylab import *

import pickle

from experiments.plotters.plotting_utils import arena_plot

def position_plot(path,α=0,ω=-1,fast=True) :
    figure()
    xlim(-5,5)
    ylim(-5,5)
    time = np.load(path+'time.npy')
    x = np.load(path+'x.npy')
    y = np.load(path+'y.npy')

    po = pickle.load(open(path+'pickle_objs.pkl','rb'))
    # DT = po['DT']
    # Ω = po['Ω']
    # β = po['β']

    training_stop_iteration = po['training_stop_iteration']

    #α,ω = len(time)-2560,len(time)
    if α < 0 :
        α = len(time)+α
    if ω < 0 :
        ω = len(time)+ω+1

    if fast :
        arena_plot(x[α:ω],y[α:ω],-5,5,-5,5,alpha=0.5,color='k')
    else :
        for σ in range(α,ω):
            #percent_complete(σ,len(time),title='Plotting Position',color='y',bar_width=30)
            if σ < training_stop_iteration :
                color = 'c'
            else :
                color = 'k'
            arena_plot(x[σ:σ+2],y[σ:σ+2],-5,5,-5,5,alpha=0.5,color=color)


    tight_layout()
    savefig(path+f'position_{α}-{ω}.png',dpi=300)
    close()