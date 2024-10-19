'''
Giulia D'Angelo, giulia.dangelo@fel.cvut.cz

This script run the Speck visualizer to visualize the DVS events.
'''



# Import the necessary module from Sinabs for interacting with Dynap-CNN devices
import sinabs.backend.dynapcnn.io as sio
# Import the Samna library, which helps build and manage event-based networks for neuromorphic devices
import samna

# List all devices connected to the PC and print them for reference
device_map = sio.get_device_map()
print(device_map)

# Open the devkit by passing the specific device name
# In this case, "speck2fdevkit:0" is the device name
devkit = sio.open_device("speck2fdevkit:0")

# Create a new event filter graph to manage the data flow
samna_graph = samna.graph.EventFilterGraph()

# Initialize a configuration for the devkit (specific to Speck2f)
devkit_config = samna.speck2f.configuration.SpeckConfiguration()

# Enable monitoring of inputs from the DVS sensor
# When set to True, events are monitored instead of being sent directly to Dynaps
devkit_config.dvs_layer.raw_monitor_enable = True

# Apply the configuration to the devkit, making sure it behaves according to the above settings
devkit.get_model().apply_configuration(devkit_config)

# Create a sequential pipeline in the Samna event filter graph
# The pipeline starts with the devkit as the event source and ends with a visualizer streamer
_, _, streamer = samna_graph.sequential([
    devkit.get_model_source_node(),  # First, source events from the devkit's DVS model
    "Speck2fDvsToVizConverter",      # Convert DVS events into a format suitable for visualization
    "VizEventStreamer"               # Stream the visualized events to the visualizer tool
])

# Set the TCP port on which the visualizer will listen for incoming event data
visualizer_port = "tcp://0.0.0.0:40000"

# Launch the visualizer in a separate process, allowing it to receive data from the specified TCP port
gui_process = sio.launch_visualizer(receiver_endpoint=visualizer_port, disjoint_process=True)

# Create another sequential pipeline, this time to handle visualizer configuration
# This configures how events are shown in the visualizer UI
visualizer_config, _ = samna_graph.sequential([
    samna.BasicSourceNode_ui_event(),  # This node generates commands to manage the visualizer UI
    streamer                           # Connects the UI commands with the DVS event streamer
])

# Specify the destination for the streamer's event data as the visualizer's TCP port
streamer.set_streamer_destination(visualizer_port)

# Wait for the visualizer to confirm it's ready to receive events
# If the visualizer is not ready, raise an error
if streamer.wait_for_receiver_count() == 0:
    raise Exception(f'Connecting to visualizer on {visualizer_port} fails.')

# Configure the visualizer with details about what kind of plot to show
# Define a plot for displaying DVS activity with specific dimensions (128x128)
plot1 = samna.ui.ActivityPlotConfiguration(image_width=128, image_height=128, title="DVS Layer", layout=[0, 0, 1, 1])

# Send the visualizer configuration to the visualizer, instructing it to display the specified plot
visualizer_config.write([
    samna.ui.VisualizerConfiguration(plots=[plot1])
])

# Start the event graph, which begins processing and visualizing the DVS events
samna_graph.start()

# Wait for user input to keep the program running
input()
