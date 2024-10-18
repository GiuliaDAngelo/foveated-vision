import numpy as np
import cv2
import sinabs.backend.dynapcnn.io as sio
import samna
import time
import threading
import random
import torch
from scipy.special import iv
import torch.nn as nn
import sinabs.layers as sl
import torchvision
import tonic
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean


def fetch_events(sink, window, drop_rate, events_lock, numevs):
    while True:
        events = sink.get_events_blocking(1000)
        if events:
            filtered_events = [event for event in events if random.random() > drop_rate]
            with events_lock:
                if filtered_events:
                    window[0, [event.y for event in filtered_events], [event.x for event in filtered_events]] = 255
                    numevs[0] += len(filtered_events)



def network_init(filters):
    """
    Initialize a neural network with a single convolutional layer using von Mises filters.

    Args:
        filters (torch.Tensor): Filters to be loaded into the convolutional layer.

    Returns:
        net (nn.Sequential): A simple neural network with one convolutional layer.
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # Define a sequential network with a Conv2D layer followed by an IAF layer
    net = nn.Sequential(
        nn.Conv2d(1, filters.shape[1], filters.shape[2], stride=1, bias=False),
        # sl.IAF()
        sl.LIF(tau_mem=tau_mem),
        # #add winner take all layer
        # WinnerTakesAll(k=1)  # Add the WTA layer here
    )
    # Load the filters into the network weights
    net[0].weight.data = filters.unsqueeze(1).to(device)
    net[1].v_mem = net[1].tau_mem * net[1].v_mem.to(device)
    return net


def run_attention(evframe):
    # Create resized versions of the frames
    resized_frames = [torchvision.transforms.Resize((int(evframe.shape[2] / pyr), int(evframe.shape[1] / pyr)))(evframe)
                      for pyr in range(1, num_pyr + 1)]
    # Process frames in batches
    batch_frames = torch.stack(
        [torchvision.transforms.Resize((resolution[0], resolution[1]))(frame) for frame in resized_frames]).type(torch.float32)
    batch_frames = batch_frames.to(device)  # Move to GPU if available
    output_rot = net(batch_frames)
    # Sum the outputs over rotations and scales
    salmap = torch.sum(torch.sum(output_rot, dim=1, keepdim=True), dim=0, keepdim=True).squeeze().type(torch.float32)
    return salmap


def create_vm_filters(thetas, size, rho, r0, thick, offset):
    """
    Create a set of Von Mises filters with different orientations.

    Args:
        thetas (np.ndarray): Array of angles in radians.
        size (int): Size of the filter.
        rho (float): Scale coefficient to control arc length.
        r0 (int): Radius shift from the center.

    Returns:
        filters (list): List of Von Mises filters.
    """
    filters = []
    for theta in thetas:
        filter = vm_filter(theta, size, rho=rho, r0=r0, thick=thick, offset=offset)
        filter = rescale(filter, fltr_resize_perc, anti_aliasing=False)
        filters.append(filter)
    filters = torch.tensor(np.stack(filters).astype(np.float32))
    return filters

def vm_filter(theta, scale, rho=0.1, r0=0, thick=0.5, offset=(0, 0)):
    """Generate a Von Mises filter with r0 shifting and an offset."""
    height, width = scale, scale
    vm = np.empty((height, width))
    offset_x, offset_y = offset

    for x in range(width):
        for y in range(height):
            # Shift X and Y based on r0 and offset
            X = (x - width / 2) + r0 * np.cos(theta) - offset_x * np.cos(theta)
            Y = (height / 2 - y) + r0 * np.sin(theta) - offset_y * np.sin(theta)  # Inverted Y for correct orientation
            r = np.sqrt(X**2 + Y**2)
            angle = zero_2pi_tan(X, Y)

            # Compute the Von Mises filter value
            vm[y, x] = np.exp(thick*rho * r0 * np.cos(angle - theta)) / iv(0, r - r0)
    # normalise value between -1 and 1
    # vm = vm / np.max(vm)
    # vm = vm * 2 - 1
    return vm

def zero_2pi_tan(x, y):
    """
    Compute the angle in radians between the positive x-axis and the point (x, y),
    ensuring the angle is in the range [0, 2π].

    Args:
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.

    Returns:
        angle (float): Angle in radians, between 0 and 2π.
    """
    angle = np.arctan2(y, x) % (2 * np.pi)  # Get the angle in radians and wrap it in the range [0, 2π]
    return angle

def plot_filters(filters, angles):
    """
    Plot the von Mises filters using matplotlib.

    Args:
        filters (torch.Tensor): A tensor containing filters to be visualized.
    """
    # Create subplots for 8 orientation VM filters
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle(f'VM filters size ({filters.shape[1]},{filters.shape[2]})', fontsize=16)

    # Display filters with their corresponding angles
    for i in range(8):
        if i < 4:
            axes[0, i].set_title(f"{round(angles[i],2)} grad")
            axes[0, i].imshow(filters[i])
            plt.colorbar(axes[0, i].imshow(filters[i]))
        else:
            axes[1, i - 4].set_title(f"{round(angles[i],2)} grad")
            axes[1, i - 4].imshow(filters[i])
            plt.colorbar(axes[1, i - 4].imshow(filters[i]))
    # add color bar to see the values of the filters
    plt.show()

if __name__ == "__main__":
    # Visualization parameters
    resolution = [128, 128]  # Resolution of the DVS sensor
    drop_rate = 0.6  # Percentage of events to drop
    update_interval = 0.02  # Update every 0.02 seconds
    # time_wnd_frames = 100000  # us ???? do I need it?

    # Visual attention paramters
    size = 10  # Size of the kernel
    r0 = 8  # Radius shift from the center
    rho = 0.1  # Scale coefficient to control arc length
    theta = np.pi * 3 / 2  # Angle to control the orientation of the arc
    thick = 3  # thickness of the arc
    offsetpxs = size/4
    offset = (offsetpxs, offsetpxs)
    fltr_resize_perc = [2, 2]
    num_pyr = 3

    # Create Von Mises (VM) filters with specified parameters
    # The angles are generated in radians, ranging from 0 to 2π in steps of π/4
    thetas = np.arange(0, 2 * np.pi, np.pi / 4)
    # Create filters using the VM function with parameters like size, rho, r0, thick, and offset
    filters = create_vm_filters(thetas, size, rho, r0, thick, offset)
    show_ftrs = False
    if show_ftrs:
        plot_filters(filters, thetas)


    # Initialize the network with the loaded filters
    tau_mem = 100  # missilecons
    net = network_init(filters)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    # run the network
    last_update_time = time.time()

    # List all connected devices
    device_map = sio.get_device_map()
    print(device_map)

    # Open the devkit device
    devkit = sio.open_device("speck2fdevkit:0")

    # Create and configure the event streaming graph
    samna_graph = samna.graph.EventFilterGraph()
    devkit_config = samna.speck2f.configuration.SpeckConfiguration()
    devkit_config.dvs_layer.raw_monitor_enable = True
    devkit.get_model().apply_configuration(devkit_config)
    sink = samna.graph.sink_from(devkit.get_model_source_node())
    samna_graph.start()
    devkit.get_stop_watch().start()
    devkit.get_stop_watch().reset()

    # Create an empty window for event visualization
    window = torch.zeros((1, resolution[1], resolution[0]), dtype=torch.uint8)
    numevs = [0]  # Use a list to allow modification within the thread
    events_lock = threading.Lock()

    # Start the event-fetching thread
    event_thread = threading.Thread(target=fetch_events, args=(sink, window, drop_rate, events_lock, numevs))
    event_thread.daemon = True
    event_thread.start()

    # Main loop for visualization
    while True:
        current_time = time.time()
        with events_lock:
            if current_time - last_update_time > update_interval:
                if numevs[0] > 0:
                    salmap = run_attention(window)
                    # cv2.imshow('DVS Events', window[0].cpu().numpy())
                    cv2.imshow('Saliency map',
                               cv2.applyColorMap(cv2.convertScaleAbs(salmap.detach().cpu().numpy()), cv2.COLORMAP_JET))
                    cv2.waitKey(1)
                    window.fill_(0)
                    numevs[0] = 0
                last_update_time = current_time