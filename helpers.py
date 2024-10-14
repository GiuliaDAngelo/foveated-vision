import os
os.environ['ApplePersistenceIgnoreState'] = 'YES'
import torch
import sinabs.layers as sl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from config import *


class RFs:
    def __init__(self, tau_mem):
        self.neuron = sl.LIF(tau_mem=tau_mem)
        self.vmem = []
        self.spikes = []

def plot_spikes(neurons):
    plt.figure()
    neuron_index = 0
    for x in range(width):
        for y in range(height):
            neuron = neurons[y][x]
            spikes = neuron.spikes
            for i in range(len(spikes)):
                plt.plot(range(len(spikes[i].flatten())), spikes[i].flatten(), '|', color='black',
                         label=f'Neuron {neuron_index}')
            neuron_index += 1
    plt.title("Spike raster")
    plt.xlabel("$t$ [ms]")
    plt.ylabel("neuron index")
    # y_axis with neurons index
    plt.yticks(range(num_neurons), range(num_neurons))
    plt.show()



def plot_mempotential(neurons, x, y):
    plt.figure()
    plt.plot(range(lenstim - 1), neurons[y][x].vmem)
    plt.title("LIF membrame dynamics")
    plt.xlabel("$t$ [ms]")
    plt.ylabel("$V_{mem}$");
    plt.show()