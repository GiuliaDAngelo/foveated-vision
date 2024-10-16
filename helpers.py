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
        self.cx = 0
        self.cy = 0
        self.radius = 0
        self.R = 0
        self.S = 0
        self.ID = 0


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


# Function to compute the size of a receptive field based on its eccentricity (rho).
# If rho is less than rho0 (the blind spot), no receptive field is generated.
def compute_rf_size(rho, W_max, rho0, R):
    if rho < rho0:
        return 0  # No receptive fields inside the blind spot
    # Scale the RF size based on the eccentricity, relative to W_max.
    rf_size = W_max * (rho / R)
    return rf_size

# Function to convert polar coordinates (rho, psi) to Cartesian coordinates (x, y).
def polar_to_cartesian(rho, psi):
    x = rho * np.cos(psi)  # Convert polar coordinate rho to Cartesian x
    y = rho * np.sin(psi)  # Convert polar coordinate psi to Cartesian y
    return x, y

# Function to rescale the rho value to ensure it fits within the defined plot dimensions.
def rescale_rho(rho, max_rho, R):
    # The rescaling ensures that the rho value remains proportional
    # while fitting within the maximum defined radius.
    return (rho / (rho0 * a ** R)) * max_rho


import numpy as np
import matplotlib.pyplot as plt


# Assuming the following parameters are defined
# a = 1.3
# q = 2
# rho0 = 0.5
# R = 10
# S = 15
# width = 128
# height = 128

def create_eccentric_RFs():
    # Initialize neurons list
    neurons = [RFs(tau_mem=tau_mem) for _ in range(R * S)]
    neuronID = 0

    # Compute W_max, the maximum receptive field size, based on Equation 4.
    W_max = rho0 * (a ** R) * (1 - a ** (-1))

    # Calculate the maximum rho value, which should be half of the diagonal of the plot area.
    max_rho = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)

    # Create an empty mask initialized to -1 (or any value to indicate no receptive field)
    mask = [[[] for _ in range(width)] for _ in range(height)]
    # Set up the plot
    fig, ax = plt.subplots()  # Create a new figure and axes
    ax.set_aspect('equal')  # Maintain equal scaling on both axes

    # Loop over the rings (eccentricity) and sectors (angular positions)
    for r in range(1, R + 1):  # Start from 1 to avoid plotting in the blind spot
        # Calculate the eccentricity (radius from the fovea) for the current ring
        rho = rho0 * (a ** r)
        # Rescale rho to fit the plot dimensions
        rescaled_rho = rescale_rho(rho, max_rho, R)

        # Loop through each sector (angular division)
        for s in range(S):  # Iterate over the number of sectors
            # Calculate the angular position for the current sector
            psi = 2 * np.pi * s / S

            # Convert polar coordinates to Cartesian coordinates for the receptive field center
            cx, cy = polar_to_cartesian(rescaled_rho, psi)
            # Shift the coordinates to center them in the plot
            cx += width // 2
            cy += height // 2

            # Compute the receptive field size based on the eccentricity
            circle_radius = int(compute_rf_size(rho, W_max, rho0, R))

            # Create LIF neuron
            neurons[neuronID].cx = cx
            neurons[neuronID].cy = cy
            neurons[neuronID].radius = circle_radius
            neurons[neuronID].R = r
            neurons[neuronID].S = s
            neurons[neuronID].ID = neuronID

            # Create a circle patch to represent the receptive field
            circle = plt.Circle((cx, cy), circle_radius, color='k', fill=False)
            ax.add_patch(circle)  # Add the circle to the axes
            # Mark the center of the receptive field with a small red dot
            plt.scatter(cx, cy, color='r', s=1)

            # Update the mask with the neuron ID for pixels inside the receptive field
            for i in range(-circle_radius, circle_radius + 1):
                for j in range(-circle_radius, circle_radius + 1):
                    if i ** 2 + j ** 2 <= circle_radius ** 2:  # Check if the pixel is inside the circle
                        x_pixel = int(cx + j)
                        y_pixel = int(cy + i)
                        # Check if pixel is within the mask boundaries
                        if 0 <= x_pixel < width and 0 <= y_pixel < height:
                            mask[y_pixel][x_pixel].append(neuronID)  # Set the neuron ID  # Set the neuron ID

            neuronID += 1

    return neurons, mask, ax

def plot_RFs(neurons, mask, ax):
    # Set plot limits and labels
    ax.set_xlim([0, width])  # Set x-axis limits to the width of the plot
    ax.set_ylim([0, height])  # Set y-axis limits to the height of the plot
    ax.set_title("Log-Polar Mapping of Retinal Receptive Fields with Eccentricity Scaling")
    ax.set_xlabel("X")  # Label for x-axis
    ax.set_ylabel("Y")  # Label for y-axis
    plt.show(block=True)

def plot_mask(mask, width, height):
    # Convert mask to a 2D array with consistent shape
    mask_array = np.full((height, width), -1, dtype=int)
    for y in range(height):
        for x in range(width):
            if mask[y][x]:
                if max(mask[y][x]) != 0:
                    mask_array[y][x] = max(mask[y][x])  # Use the maximum neuron ID

    # Plot the mask
    plt.imshow(mask_array, cmap='jet', interpolation='nearest')  # Display the mask with a colormap
    plt.colorbar(label='Neuron ID')  # Add a color bar to indicate neuron IDs
    plt.title('Neuron ID Mask')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show(block=True)
