from model import Model
from experiments.light_experiment import LightExperiment
from experiments.oscillation_experiment import OscillationExperiment
from oscillation_body import OscillationBody
from experiments.pattern_experiment import NoTrainingExperiment, PatternExperiment

from pattern_body import PatternBody
from braitenberg_body import BraitenbergBody
import numpy as np
from threading import Thread

from body import Body
from brain import Brain
from world import BraitenbergWorld, EmptyWorld
from experiments.experiment import Experiment

import cProfile

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("simple_example")
    #parser.add_argument("counter", help="An integer will be increased by 1 and printed.", type=int)
    parser.add_argument("--headless", help="Run in headless mode.", action="store_true")
    parser.add_argument("--experiment", help="Experiment to run.", type=str)
    parser.add_argument("--seed", help="RNG seed", type=int, default=np.random.randint(0,9))
    
    args = parser.parse_args()
    
    if args.experiment == 'pattern':
        experiment = PatternExperiment
    if args.experiment == 'no_training':
        experiment = NoTrainingExperiment
    elif args.experiment == 'oscillation':
        experiment = OscillationExperiment
    elif args.experiment == 'light':
        experiment = LightExperiment
    else :
        experiment = PatternExperiment
    
    ## create model
    if args.headless:
        m = Model(headless=True, experiment_class=experiment,seed=args.seed)        

    if not args.headless:
        ## create headful model
        m = Model(headless=False, experiment_class=experiment,seed=args.seed)        
        ## attach RVIT head
        from rvit_wrapper import attach_rvit
        attach_rvit(m)

        
