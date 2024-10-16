import sinabs.backend.dynapcnn.io as sio
import samna
import time

# List all connected devices that are currently plugged into the PC.
# This helps identify the devices available for communication.
device_map = sio.get_device_map()
print(device_map)

# Open the devkit device by specifying its name (e.g., "speck2fdevkit:0").
# This initializes the communication with the specific device.
devkit = sio.open_device("speck2fdevkit:0")

# Optionally, you could enable a timestamping feature for events (currently commented out).
# devkit.set_timestamper_enable(True)

# Create an EventFilterGraph instance from samna, which will be used to handle event streaming.
samna_graph = samna.graph.EventFilterGraph()

# Initialize a SpeckConfiguration object, which defines the configuration settings for the device.
devkit_config = samna.speck2f.configuration.SpeckConfiguration()

# Enable the raw monitor for the DVS (Dynamic Vision Sensor) layer.
# This setting allows us to monitor the DVS input events before they are sent to the DynapCNN chip.
devkit_config.dvs_layer.raw_monitor_enable = True  # If set to False, events will be processed by DynapCNN directly.

# Apply the configuration to the devkit model.
# This step ensures that the device operates according to the specified settings.
devkit.get_model().apply_configuration(devkit_config)

# Create a sink node, which will act as the data destination in the event stream.
# The sink is connected to the source node of the model, where the events are generated.
sink = samna.graph.sink_from(devkit.get_model_source_node())

# Start the samna event processing graph to begin handling the event stream.
samna_graph.start()

# Fetch a block of events from the sink node, with a timeout of 1000ms (1 second).
# This blocks the execution until events are available or the timeout occurs.
events = sink.get_events_blocking(1000)

# Enable and reset the stopwatch on the devkit.
# The stopwatch can be useful for performance measurements or event timing.
devkit.get_stop_watch().set_enable_value(True)
devkit.get_stop_watch().reset()

# Continuously process events in an infinite loop.
while True:
    # Fetch another block of events from the sink, using a timeout of 1000ms.
    # These events would normally be processed by DynapCNN, unless raw_monitor_enable is set to True.
    events = sink.get_events_blocking(1000)

    # Print the retrieved events for debugging or analysis purposes.
    print(events)
