#### POSITION BY TIME SLICES PLOT
import pickle
from experiments.plotters.plotting_utils import arena_plot
from pylab import *


def position_time_slices_plot(path) :
    time = np.load(path+'time.npy')
    x = np.load(path+'x.npy')
    y = np.load(path+'y.npy')
    po = pickle.load(open(path+'pickle_objs.pkl','rb'))
    training_stop_iteration = po['training_stop_iteration']
    
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
        if α < training_stop_iteration :
            color = 'r'
        else :
            color = 'k'
        arena_plot(x[α:ω],y[α:ω],-5,5,-5,5,color=color)
        xticks([])
        yticks([])

    tight_layout()
    savefig(path+'position_time_slices.png',dpi=300)