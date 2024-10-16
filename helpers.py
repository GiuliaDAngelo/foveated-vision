import os
os.environ['ApplePersistenceIgnoreState'] = 'YES'
import torch
import sinabs.layers as sl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from config import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# This is a fuction to simplify a fraction


class RFs:
    def __init__(self, tau_mem):
        self.neuron = sl.LIF(tau_mem=tau_mem)
        self.vmem = []
        self.spikes_ts = []


import matplotlib.pyplot as plt


def plot_spikes(neurons):
    spike_trains = []
    plt.figure()

    neuron_index = 0
    for x in range(width):
        for y in range(height):
            spikes = neurons[y][x].spikes_ts  # Extract spike train for current neuron
            spike_times = np.zeros(lenstim)  # Initialize an array of zeros
            spike_times[spikes] = 1  # Set positions of spikes to 1
            spike_trains.append(spike_times)  # Add to spike_trains list

            # Plot vlines at times where spikes occurred (spike == 1) for each neuron
            plt.vlines(np.where(spike_times == 1)[0], ymin=neuron_index - 1, ymax=neuron_index + 1, color='black',
                       linewidth=0.8)

        neuron_index += 1

    # Plot formatting
    plt.title("Spike Raster Plot")
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron Index")
    plt.xlim(0, lenstim)
    plt.ylim(-0.5, neuron_index - 0.5)
    plt.show()



def plot_mempotential(neurons, x, y):
    plt.figure()
    plt.plot(range(lenstim - 1), neurons[y][x].vmem)
    plt.title("LIF membrame dynamics")
    plt.xlabel("$t$ [ms]")
    plt.ylabel("$V_{mem}$");
    plt.show()


