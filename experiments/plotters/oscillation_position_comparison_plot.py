import pickle
from pylab import *

def position_plot_range(path,α=0,ω=1024*4) :
    time = np.load(path+'time.npy')
    prediction_error = np.load(path+'prediction_error.npy')
    x = np.load(path+'x.npy')
    po = pickle.load(open(path+'pickle_objs.pkl','rb'))
    β = po['β']
    
    plot(time[α:ω],x[α:ω],'-',color='k',lw=0.5,label='x')
    minv = prediction_error[β+1:].min()
    maxv = prediction_error[β+1:].max()
    ylabel('x',rotation=0)
    xlabel('time')
    xlim(time[α],time[ω])

def oscillation_position_comparison_plot(path) :
    figure(figsize=(7,2.5))

    subplot2grid((2,1),(0,0))
    position_plot_range(path,α=0,ω=1024*4)
    xlabel('')

    subplot2grid((2,1),(1,0))
    position_plot_range(path,α=-1024*4-2,ω=-2)

    tight_layout()
    savefig(path+'osc_position_comparison.png',dpi=300)