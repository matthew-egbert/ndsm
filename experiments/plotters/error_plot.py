import pickle
from matplotlib import patches
from pylab import *


def error_plot(path,save_as_fig=True) :
    po = pickle.load(open(path+'pickle_objs.pkl','rb'))
    DT = po['DT']
    Ω = po['Ω']
    β = po['β']
    training_stop_iteration = po['training_stop_iteration']

    time = np.load(path+'time.npy')
    prediction_error = np.load(path+'prediction_error.npy')

    figure(figsize=(7,2.5))
    plot(time[β+1:],prediction_error[β+1:],label='log($\\epsilon$)',color='k',lw=0.5)
    minv = prediction_error[β+1:].min()
    maxv = prediction_error[β+1:].max()
    if training_stop_iteration > 1 :
        rect = patches.Rectangle((0,minv), training_stop_iteration*DT, maxv+0.4, linewidth=0, edgecolor='w', facecolor='0.666')
        ax = gca()                
        text(0.01,0.08, 'TRAINING PHASE', fontsize=6, ha='left', color='w', transform=ax.transAxes)
        gca().add_patch(rect)
    yscale('log')
    ylabel('log($\\epsilon$)')
    xlabel('time')
    xlim(0,time[-1])
    tight_layout()
    plt.box(False)
    if save_as_fig:
        savefig(path+'error.png',dpi=300)
        #close()
    