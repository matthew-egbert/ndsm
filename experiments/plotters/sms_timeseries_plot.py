import pickle
from matplotlib import patches
from pylab import *


def sms_timeseries_plot(path,α=0,ω=-1) :        
    po = pickle.load(open(path+'pickle_objs.pkl','rb'))
    DT = po['DT']
    Ω = po['Ω']
    β = po['β']
    
    time = np.load(path+'time.npy')
    sms = np.load(path+'sms.npy')
    #prediction_error = np.load(path+'prediction_error.npy')

     
    if α < 0 :
        α = len(time)+α
    if ω < 0 :
        ω = len(time)+ω
    
    dim = sms.shape[1]
    figure(figsize=(7,2.5*dim))
    for row in range(dim) :
        subplot2grid((dim,1),(row,0))
        plot(time[α:ω],sms[α:ω,row],color='k',lw=1)        
        xlim(time[α],time[ω])
        #ylim(-10,10)        
        plt.box(False)
    
    tight_layout()
    
    #if save_as_fig:
    savefig(path+'sms_timeseries.png',dpi=300)
    close()
    