import pickle
from pylab import *
from plotters.oscillation_network_plot import oscillation_network_plots
from plotters.error_plot import error_plot
from plotters.position_time_slices_plot import position_time_slices_plot
from plotters.pattern_publication_plots import pattern_publication_plots
from plotters.sms_slice_plot import sms_slice_plot
from plotters.position_plot import position_plot
  
def analyse_pattern(path) :    
    #error_plot(path) #a special one is made in publication plots...
    #position_plot(path)
    #position_time_slices_plot(path)
    pattern_publication_plots(path)

def analyse_oscillation(path) :
    oscillation_network_plots(path)

def analyse(path) :
    po = pickle.load(open(path+'pickle_objs.pkl','rb'))
    if po['experiment'] == 'PatternExperiment' :
        analyse_pattern(path)
    elif po['experiment'] == 'OscillationExperiment' :
        analyse_oscillation(path)
    else :
        print('Unknown experiment type')

if __name__ == '__main__' :
    path = 'results/PatternExperiment_8/'
    analyse_pattern(path)

    
            
    


    
        

