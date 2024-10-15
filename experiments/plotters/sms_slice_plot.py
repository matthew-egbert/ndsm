import pickle
from pylab import *

def sms_slice_plot(path) :
    sms = np.load(path+'sms.npy')
    time = np.load(path+'time.npy')
    po = pickle.load(open(path+'pickle_objs.pkl','rb'))
    training_stop_iteration = po['training_stop_iteration']
    def cleanplot() :
            xticks([])
            yticks([])
            plt.box(False)
            xlim(-1.1,1.1)
            ylim(-0.1,1.1)

    #### SENSORIMOTOR SPACE BY TIMESLICE
    figure(figsize=(12,15))
    R = 8
    C = 8
    N = R*C
    section_length = len(time)//N
    for i in range(N):
        subplot2grid((R,C),(i//C,i%C))
        α = i*section_length
        ω = (i+1)*section_length

        for ls_i in range(α,ω-2):
            if α < training_stop_iteration :
                color = 'r'
            else :
                color = 'k'
            μ = 0.03
            plot(sms[ls_i:ls_i+2,1]+np.random.randn(2)*μ,
                sms[ls_i:ls_i+2,0]+np.random.randn(2)*μ,alpha=0.2,lw=2.0,color=color)
        cleanplot()

    tight_layout()
    savefig(path+'sms.png',dpi=300)