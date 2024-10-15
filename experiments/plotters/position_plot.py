from pylab import *

import pickle

from experiments.plotters.plotting_utils import arena_plot

def position_plot(path) :
    time = np.load(path+'time.npy')
    x = np.load(path+'x.npy')
    y = np.load(path+'y.npy')

    po = pickle.load(open(path+'pickle_objs.pkl','rb'))
    # DT = po['DT']
    # Ω = po['Ω']
    # β = po['β']

    training_stop_iteration = po['training_stop_iteration']

    α,ω = len(time)-2560,len(time)

    for σ in range(α,ω):
            #percent_complete(σ,len(time),title='Plotting Position',color='y',bar_width=30)
            if σ < training_stop_iteration :
                color = 'c'
            else :
                color = 'k'
            arena_plot(x[σ:σ+2],y[σ:σ+2],-5,5,-5,5,alpha=0.5,color=color)
            #arena_plot(x[σ:σ+step],y[σ:σ+step],alpha=0.1,color=color)

    tight_layout()
    savefig(path+'position_full.png',dpi=300)
    close()