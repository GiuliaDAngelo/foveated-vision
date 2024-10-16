'''
Giulia D'Angelo, giulia.dangelo@fel.cvut.cz

This script simulate the eccentric representation of the retina using log-polar mapping.
The code does take isnpiration from the following article:

Chessa, M., Maiello, G., Bex, P. J., & Solari, F. (2016). A space-variant model for motion interpretation across the visual field. Journal of vision, 16(2), 12-12.
'''


import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from config import *


if __name__ == "__main__":
    # Compute W_max, the maximum receptive field size, based on Equation 4.
    # This value will help determine the scaling of the receptive fields across eccentricities.
    W_max = rho0 * (a ** R) * (1 - a ** (-1))

    # Calculate the maximum rho value, which should be half of the diagonal of the plot area.
    # This ensures that the receptive fields fit within the defined plot dimensions.
    max_rho = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)


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

            # Convert the polar coordinates to Cartesian coordinates for the receptive field center
            cx, cy = polar_to_cartesian(rescaled_rho, psi)
            # Shift the coordinates to center them in the plot
            cx += width // 2
            cy += height // 2

            # Compute the receptive field size based on the eccentricity
            circle_radius = compute_rf_size(rho, W_max, rho0, R)

            # Create a circle patch to represent the receptive field
            circle = plt.Circle((cx, cy), circle_radius, color='k', fill=False)
            ax.add_patch(circle)  # Add the circle to the axes
            # Mark the center of the receptive field with a small red dot
            plt.scatter(cx, cy, color='r', s=1)

    # Set plot limits and labels
    ax.set_xlim([0, width])  # Set x-axis limits to the width of the plot
    ax.set_ylim([0, height])  # Set y-axis limits to the height of the plot
    ax.set_title("Log-Polar Mapping of Retinal Receptive Fields with Eccentricity Scaling")
    ax.set_xlabel("X")  # Label for x-axis
    ax.set_ylabel("Y")  # Label for y-axis

    # Display the plot
    plt.show()
