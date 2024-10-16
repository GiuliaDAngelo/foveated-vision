import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 1.3  # Nonlinearity parameter for log-polar mapping. This controls how the receptive field sizes change with eccentricity.
q = 2    # Scaling factor for eta, which represents angular displacement in the polar coordinates.
rho0 = 0.5  # Radius of the central blind spot, where no receptive fields are generated.
R = 14   # Number of rings (eccentricity levels) representing different distances from the fovea.
S = 15   # Number of sectors (angular divisions) that split the visual field around the fovea.
width = 128  # Width of the plot area in pixels.
height = 128  # Height of the plot area in pixels.

# Compute W_max, the maximum receptive field size, based on Equation 4.
# This value will help determine the scaling of the receptive fields across eccentricities.
W_max = rho0 * (a ** R) * (1 - a ** (-1))

# Calculate the maximum rho value, which should be half of the diagonal of the plot area.
# This ensures that the receptive fields fit within the defined plot dimensions.
max_rho = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)

# Function to compute the size of a receptive field based on its eccentricity (rho).
# If rho is less than rho0 (the blind spot), no receptive field is generated.
def compute_rf_size(rho, a, rho0, R):
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


if __name__ == "__main__":
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
            circle_radius = compute_rf_size(rescaled_rho, a, rho0, R)

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
