
'''

Giulia D'Angelo, giulia.dangelo@fel.cvut.cz

This script simulates RFs in the retina using Leaky Integrate-and-Fire (LIF) neurons.
It generates a Poisson-distributed spike train and processes it through a 2D grid of LIF neurons.
The script visualizes the spike raster and membrane potential dynamics of the neurons over time.

Look for the config file to set the parameters of the simulation (e.g., tau_mem, width, height, etc.).
Functions are in helpers.py.

'''

import sys
sys.path.append('/Users/giuliadangelo/workspace/code/foveated-vision')

import os
os.environ['ApplePersistenceIgnoreState'] = 'YES'  # Ensures smooth interaction with Apple’s window state persistence
import torch
import sinabs.layers as sl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from helpers.helpers import *  # Import helper functions (likely custom utility code)
from helpers.config import *  # Import configuration parameters (e.g., tau_mem, width, height, etc.)

matplotlib.use('qt5agg')  # Sets up Matplotlib to use the Qt5 backend for graphical display

# Generate a Poisson-distributed tensor representing a spike train over 'lenstim' time steps
rate_spikes = 5
ts = torch.poisson(torch.ones(1, lenstim, 1) * rate_spikes).float()  # Increase the rate parameter to 5

# Initialize arrays to hold the x, y coordinates and 'p' (some other parameter, possibly neuron activity or position)
x = np.random.randint(0, width, lenstim)
y = np.random.randint(0, height, lenstim)
p = np.zeros(lenstim).astype(int)

# Create a 2D grid (height x width) of receptive field neurons (RFs), with each neuron having membrane properties 'tau_mem'
neurons = [[RFs(tau_mem=tau_mem) for _ in range(width)] for _ in range(height)]

# Output spike raster for visualization
with torch.no_grad():  # Disable gradient computation (since it's not needed for this operation)
    for t in range(0, lenstim - 1):  # Loop over each time step
        # Select the receptive field neuron at the current (x, y) location
        out = neurons[y[t]][x[t]].neuron(ts[:, t: t + 1])
        # Check if the neuron produced any spikes
        if (out != 0).any():
            # If so, record the spike by appending the spike data
            neurons[y[t]][x[t]].spikes_ts.append(t)
        # Record the neuron's membrane potential at the current time step
        neurons[y[t]][x[t]].vmem.append(neurons[y[t]][x[t]].neuron.v_mem[0, 0])


# Plot spike raster from neurons over time
plot_spikes(neurons)

# Plot membrane potential of the neuron at (0, 0) location over time
plot_mempotential(neurons, 0, 0)

print('end')  # Signal the end of the script execution
