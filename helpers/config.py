


# camera parameters for Speck
width = 128
height = 128

# parameters RFs
num_neurons = width * height
tau_mem = 1

#parameters events
lenstim = 1000
t_window = 20

#parameters eccentric retina
a = 1.3 # Nonlinearity parameter for log-polar mapping. This controls how the receptive field sizes change with eccentricity.
q = 2    # Scaling factor for eta, which represents angular displacement in the polar coordinates.
rho0 = 0.5  # Radius of the central blind spot, where no receptive fields are generated.
R = 16   # Number of rings (eccentricity levels) representing different distances from the fovea.
S = 24   # Number of sectors (angular divisions) that split the visual field around the fovea.
