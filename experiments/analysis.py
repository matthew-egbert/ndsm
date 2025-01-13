import pickle
from pylab import *
from experiments.plotters.sms_timeseries_plot import sms_timeseries_plot
from experiments.plotters.position_plot import position_plot
from experiments.plotters.oscillation_position_comparison_plot import oscillation_position_comparison_plot
from experiments.plotters.oscillation_network_plot import oscillation_network_plots
from experiments.plotters.error_plot import error_plot
from experiments.plotters.position_time_slices_plot import position_time_slices_plot
from experiments.plotters.pattern_publication_plots import pattern_publication_plots

import numpy as np

def longest_repeating_pattern(sequence, max_length):
    # Convert sequence to a numpy array if it's not already
    sequence = np.array(sequence)
    
    # Calculate the autocorrelation
    autocorr = correlate(sequence, sequence, mode='full')
    autocorr = autocorr[autocorr.size // 2:]  # Keep only the second half
    plot(autocorr)
    figure()
    # show()

    autocorr[0] = 0  # Ignore the peak at 0
    # Find the peaks in the autocorrelation, indicating periodic repeats
    #peaks = np.where(autocorr > 0)[0]
    peaks = [np.argmax(autocorr),]
    longest_pattern = None

    # Check each peak to find repeating subsequences shorter than max_length
    for peak in peaks:
        if peak >= max_length:
            break
        pattern = sequence[:peak]  # Potential repeating pattern
        # Verify it repeats
        if np.array_equal(sequence[:peak], sequence[peak:2*peak]):
            longest_pattern = pattern

    return longest_pattern if longest_pattern is not None else []

def find_pattern(path) :
    po = pickle.load(open(path+'pickle_objs.pkl','rb'))
    DT = po['DT']
    Ω = po['Ω']
    β = po['β']
    time = np.load(path+'time.npy')
    prediction_error = np.load(path+'prediction_error.npy')
    sms = np.load(path+'sms.npy')
    
    tail_start_index = -1024
    sequence = sms[tail_start_index:,0]*10 + sms[tail_start_index:,1]
    #print(sequence)
    #quit()
    #show()
    #print('not the actual sequence?')
    x = longest_repeating_pattern(sequence,Ω)
    plot(x)
    print(len(x))
    show()
    
    
    


def analyse_pattern(path) :    
    error_plot(path) #a special one is made in publication plots...
    position_plot(path,α=-2560,ω=-1)
    position_time_slices_plot(path)
    pattern_publication_plots(path)

def analyse_no_training(path) :
    # error_plot(path)
    # position_plot(path,α=-256,ω=-1)
    # position_time_slices_plot(path)
    # #pattern_publication_plots(path)
    find_pattern(path)

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
    elif po['experiment'] == 'NoTrainingExperiment' :
        analyse_no_training(path)
    else :
        print('Unknown experiment type')

if __name__ == '__main__' :
    ## just for testing
    path = 'results/NoTrainingExperiment_0/'
    #path = 'results/PatternExperiment_8/'
    analyse(path)

    
            
    


    
        

