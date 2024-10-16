import sys
sys.path.append('/Users/giuliadangelo/workspace/code/foveated-vision')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
# from StereoAedatReader import *
from helpers.helpers import *
from helpers.config import *
import imageio


matplotlib.use('qt5agg')  # Sets up Matplotlib to use the Qt5 backend for graphical display

# Generate a Poisson-distributed tensor representing a spike train over 'lenstim' time steps
rate_spikes = 5
ts = torch.poisson(torch.ones(1, lenstim, 1) * rate_spikes).float()  # Increase the rate parameter to 5

# Initialize arrays to hold the x, y coordinates and 'p' (some other parameter, possibly neuron activity or position)
x = np.random.randint(0, width, lenstim)
y = np.random.randint(0, height, lenstim)
p = np.zeros(lenstim).astype(int)

# Create a 2D grid (ringsxsectors) of receptive field neurons (RFs), with each neuron having membrane properties 'tau_mem'
[neurons, mask, ax] = create_eccentric_RFs()
# plot_RFs(neurons, mask, ax)
# plot_mask(mask, width, height)
print('Number of neurons: ', len(neurons))

# create a window matrix to store and plot spikes
plt.figure()
window = np.zeros((height+1,width+1))
windows = []
tw = 20

# Output spike raster for visualization
with torch.no_grad():  # Disable gradient computation (since it's not needed for this operation)
    for t in range(0, lenstim - 1):  # Loop over each time step
        # extract non repeted mask values in x, y
        IDs = mask[y[t]][x[t]]
        for ID in IDs:
            # Select the receptive field neuron at the current (x, y) location
            out = neurons[ID].neuron(ts[:, t: t + 1])/(neurons[ID].radius**2)
            # Check if the neuron produced any spikes
            if (out != 0).any():
                # If so, record the spike by appending the spike data
                neurons[ID].spikes_ts.append(t)
                gaussian_plot(neurons, ID, window)
                if t > t_window:
                    plt.imshow(window, cmap='jet')
                    #save window as GIF
                    windows.append(np.copy(window))
                    plt.draw()
                    plt.pause(0.0001)
                    # empty window
                    window = np.zeros((height+1, width+1))
                    t_window = t + tw
            # Record the neuron's membrane potential at the current time step
            neurons[ID].vmem.append(neurons[ID].neuron.v_mem[0, 0])

# save windows as GIF
imageio.mimsave('windows.gif', windows, duration=0.1)



