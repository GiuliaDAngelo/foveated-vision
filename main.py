import os
os.environ['ApplePersistenceIgnoreState'] = 'YES'  # Ensures smooth interaction with Apple’s window state persistence
import torch
import sinabs.layers as sl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from helpers import *  # Import helper functions (likely custom utility code)
from config import *  # Import configuration parameters (e.g., tau_mem, width, height, etc.)

matplotlib.use('qt5agg')  # Sets up Matplotlib to use the Qt5 backend for graphical display

# Generate a Poisson-distributed tensor representing a spike train over 'lenstim' time steps
ts = torch.poisson(torch.ones(1, lenstim, 1)).float()

# Initialize arrays to hold the x, y coordinates and 'p' (some other parameter, possibly neuron activity or position)
x = np.zeros(lenstim).astype(int)
y = np.zeros(lenstim).astype(int)
p = np.zeros(lenstim).astype(int)

# Create a 2D grid (height x width) of receptive field neurons (RFs), with each neuron having membrane properties 'tau_mem'
neurons = [[RFs(tau_mem=tau_mem) for _ in range(width)] for _ in range(height)]

# Output spike raster for visualization
i = 0  # Initialize index counter
for t in range(0, lenstim - 1):  # Loop over each time step
    with torch.no_grad():  # Disable gradient computation (since it's not needed for this operation)
        # Select the receptive field neuron at the current (x, y) location
        RF = neurons[y[i]][x[i]]
        # Feed the current time slice of the spike train to the neuron's forward pass
        out = RF.neuron(ts[:, i: i + 1])

        # Check if the neuron produced any spikes
        if (out != 0).any():
            # If so, record the spike by appending the spike data
            RF.spikes.append(ts.unsqueeze(dim=0))

        # Record the neuron's membrane potential at the current time step
        RF.vmem.append(RF.neuron.v_mem[0, 0])

    i += 1  # Move to the next index for the next time step

# Plot spike raster from neurons over time
plot_spikes(neurons)

# Plot membrane potential of the neuron at (0, 0) location over time
plot_mempotential(neurons, 0, 0)

print('end')  # Signal the end of the script execution
