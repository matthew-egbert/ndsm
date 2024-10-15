import pickle
from pylab import *
from experiments.plotters.sms_timeseries_plot import sms_timeseries_plot
from experiments.plotters.position_plot import position_plot
from experiments.plotters.oscillation_position_comparison_plot import oscillation_position_comparison_plot
from experiments.plotters.oscillation_network_plot import oscillation_network_plots
from experiments.plotters.error_plot import error_plot
from experiments.plotters.position_time_slices_plot import position_time_slices_plot
from experiments.plotters.pattern_publication_plots import pattern_publication_plots

def analyse_pattern(path) :    
    error_plot(path) #a special one is made in publication plots...
    position_plot(path,α=-2560,ω=-1)
    position_time_slices_plot(path)
    pattern_publication_plots(path)

def analyse_oscillation(path) :
    error_plot(path)
    oscillation_position_comparison_plot(path)
    oscillation_network_plots(path)

def analyse_light(path) :
    error_plot(path)
    time = np.load(path+'time.npy')
    tot = len(time)
    α = 0#int(tot*0.5)
    ω = -1#int(tot*0.5)
    position_plot(path,α,ω)
    sms_timeseries_plot(path,α,ω)

    
def analyse(path) :
    po = pickle.load(open(path+'pickle_objs.pkl','rb'))
    if po['experiment'] == 'PatternExperiment' :
        analyse_pattern(path)
    elif po['experiment'] == 'OscillationExperiment' :
        analyse_oscillation(path)
    elif po['experiment'] == 'LightExperiment' :
        analyse_light(path)
    else :
        print('Unknown experiment type')

if __name__ == '__main__' :
    ## just for testing
    path = 'results/LightExperiment_0/'
    analyse(path)

    
            
    


    
        

