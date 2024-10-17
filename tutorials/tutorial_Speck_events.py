import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('qt5agg')
import sinabs.backend.dynapcnn.io as sio
import samna
import time

# Parameters visualisation
tw = 1000

# List all connected devices that are currently plugged into the PC.
device_map = sio.get_device_map()
print(device_map)

# Open the devkit device by specifying its name (e.g., "speck2fdevkit:0").
devkit = sio.open_device("speck2fdevkit:0")

# Create an EventFilterGraph instance from samna, which will be used to handle event streaming.
samna_graph = samna.graph.EventFilterGraph()

# Initialize a SpeckConfiguration object, which defines the configuration settings for the device.
devkit_config = samna.speck2f.configuration.SpeckConfiguration()
resolution = [128, 128]

# Enable the raw monitor for the DVS (Dynamic Vision Sensor) layer.
devkit_config.dvs_layer.raw_monitor_enable = True

# Apply the configuration to the devkit model.
devkit.get_model().apply_configuration(devkit_config)

# Create a sink node, which will act as the data destination in the event stream.
sink = samna.graph.sink_from(devkit.get_model_source_node())

# Start the samna event processing graph to begin handling the event stream.
samna_graph.start()

# Fetch a block of events from the sink node, with a timeout of 1000ms (1 second).
events = sink.get_events_blocking(1000)


# Enable and reset the stopwatch on the devkit.
devkit.get_stop_watch().set_enable_value(True)
devkit.get_stop_watch().reset()

# Create a window to plot the events in real-time
window = np.zeros((resolution[1], resolution[0]), dtype=int)
# Set up the plot
fig, ax = plt.subplots()
im = ax.imshow(window, cmap='gray', animated=True)
plt.show(block=False)

# Continuously process events in an infinite loop.
last_update_time = time.time()
update_interval = 0.1  # Update every 0.1 seconds

numevs=0
while True:
    # print("Fetching events...")
    events = sink.get_events_blocking(1000)
    if events:
        # print(f"Number of events fetched: {len(events)}")
        window[[event.y for event in events], [event.x for event in events]] = 255
        numevs+=len(events)
        if numevs>10000:
            plt.imshow(window, cmap='gray')
            plt.draw()
            plt.pause(0.0001)
            # print(window)
            window = np.zeros((resolution[1], resolution[0]), dtype=int)
            numevs=0
