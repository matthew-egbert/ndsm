#!/bin/bash
python3 -m cProfile -o profile.prof main.py
#python3 -m line_profiler main.py -- --headless --experiment pattern > profiling.output^C
