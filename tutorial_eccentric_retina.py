import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
# from StereoAedatReader import *
from helpers import *
from config import *


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
plot_RFs(neurons, mask, ax)
plot_mask(mask, width, height)

print('Number of neurons: ', len(neurons))



