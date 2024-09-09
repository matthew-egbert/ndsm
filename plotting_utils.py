import numpy as np
from pylab import *
from pylab import gca, np, plot, xlim, ylim 

def percent_complete(step, total_steps, bar_width=60, title="", print_perc=True, color='c'):
    import sys

    # UTF-8 left blocks: 1, 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8
    utf_8s = ["█", "▏", "▎", "▍", "▌", "▋", "▊", "█"]
    perc = 100 * float(step) / float(total_steps)
    max_ticks = bar_width * 8
    num_ticks = int(round(perc / 100 * max_ticks))
    full_ticks = num_ticks / 8      # Number of full blocks
    part_ticks = num_ticks % 8      # Size of partial block (array index)
    
    disp = bar = ""                 # Blank out variables
    bar += utf_8s[0] * int(full_ticks)  # Add full blocks into Progress Bar
    
    # If part_ticks is zero, then no partial block, else append part char
    if part_ticks > 0:
        bar += utf_8s[part_ticks]
    
    # Pad Progress Bar with fill character
    bar += "▒" * int((max_ticks/8 - float(num_ticks)/8.0))
    
    if len(title) > 0:
        disp = title + ": "         # Optional title to progress display
    
    # Print progress bar in green: https://stackoverflow.com/a/21786287/6929343
    if color == 'g':
        disp += "\x1b[0;32m"            # Color Green
    elif color == 'c':
        disp += "\x1b[0;36m"            # Color Cyan
    elif color == 'y':
        disp += "\x1b[0;33m"            # Color Yellow
    disp += bar                     # Progress bar to progress display
    disp += "\x1b[0m"               # Color Reset
    if print_perc:
        # If requested, append percentage complete to progress display
        if perc > 100.0:
            perc = 100.0            # Fix "100.04 %" rounding error
        disp += " {:6.2f}".format(perc) + " %"
    
    # Output to terminal repetitively over the same line using '\r'.
    sys.stdout.write("\r" + disp)
    sys.stdout.flush()


def arena_plot(x, y, x_min, x_max, y_min, y_max, sensitivity=0.9, **kwargs):
    """
    #PERIODIC #WRAP #WRAPAROUND #PLOT #PLOTTING #VISUALIZATION

    Plots data with wraparound on both axes, removing unwanted lines by interpolating
    the data to the edges of the plot and inserting NaNs to break the line.

    Parameters:
    - x: array-like, x-values of the data
    - y: array-like, y-values of the data
    - x_min: float, minimum x value where wrapping occurs
    - x_max: float, maximum x value where wrapping occurs
    - y_min: float, minimum y value where wrapping occurs
    - y_max: float, maximum y value where wrapping occurs

    Returns:
    - None (displays the plot)
    """

    x_new, y_new = [], []

    def add_segment(x1, y1, x2, y2):
        """Helper function to add segments to the new lists."""
        x_new.extend([x1, x2, np.nan])
        y_new.extend([y1, y2, np.nan])

    for i in range(len(x) - 1):
        x1, y1 = x[i], y[i]
        x2, y2 = x[i + 1], y[i + 1]

        # Check for wraparound in x
        if np.abs(x2 - x1) > (x_max-x_min)*sensitivity :
            my = (y2+y1)/2
            if x2 > x1:  # wrap around the right edge
                # plot(x1, y1, 'ro')
                # plot(x1, y2, 'rx')
                my = (y2+y1)/2
                add_segment(x1, y1, x_min, my)
                add_segment(x_max, my, x2, y2)
            else:  # wrap around the left edge
                # plot(x1, y1, 'ro')
                # plot(x1, y2, 'rx')                
                add_segment(x1, y1, x_max, my)
                add_segment(x_min, my, x2, y2)
        # Check for wraparound in y
        elif np.abs(y2 - y1) > (y_max-y_min)*sensitivity :
            mx = (x2+x1)/2
            if y2 > y1:  # wrap around the top edge
                add_segment(x1, y1, mx, y_min)
                add_segment(mx, y_max, x2, y2)
            else:  # wrap around the bottom edge
                pass
                add_segment(x1, y1, mx, y_max)
                add_segment(mx, y_min, x2, y2)
        else:
            add_segment(x1, y1, x2, y2)

    # Append the last point
    x_new.append(x[-1])
    y_new.append(y[-1])

    if 'color' not in kwargs:
        kwargs['color'] = 'k'
    plot(x_new, y_new,lw=0.8,**kwargs)
    xlim(x_min, x_max)
    ylim(y_min, y_max)
    gca().set_aspect('equal')


def running_average(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')