import numpy as np
import cv2
import sinabs.backend.dynapcnn.io as sio
import samna
import time
import threading
import random

# Visualization parameters
resolution = [128, 128]  # Resolution of the DVS sensor
drop_rate = 0.7  # Percentage of events to drop (e.g., 70% of events)

# List all connected devices
device_map = sio.get_device_map()
print(device_map)

# Open the devkit device
devkit = sio.open_device("speck2fdevkit:0")

# Create and configure the event streaming graph
samna_graph = samna.graph.EventFilterGraph()

# Initialize and apply the device configuration
devkit_config = samna.speck2f.configuration.SpeckConfiguration()
devkit_config.dvs_layer.raw_monitor_enable = True  # Enable raw monitor for DVS
devkit.get_model().apply_configuration(devkit_config)

# Create a sink node to receive event stream data
sink = samna.graph.sink_from(devkit.get_model_source_node())

# Start event processing
samna_graph.start()

# Enable and reset the devkit stopwatch
devkit.get_stop_watch().set_enable_value(True)
devkit.get_stop_watch().reset()

# Create an empty window (128x128) for event visualization
window = np.zeros((resolution[1], resolution[0]), dtype=np.uint8)

# Time interval for updating the visualization
update_interval = 0.03  # seconds
last_update_time = time.time()

# Initialize variables to manage events
numevs = 0  # Event counter
events_lock = threading.Lock()  # To ensure thread-safe updates to shared variables

# Function to fetch events in a separate thread
def fetch_events():
    global numevs
    while True:
        events = sink.get_events_blocking(1000)  # Fetch events with a timeout of 1 second
        if events:
            # Drop some events to reduce load (keep (1 - drop_rate)% of events)
            filtered_events = [event for event in events if random.random() > drop_rate]

            with events_lock:  # Lock to prevent data race conditions
                # Only process filtered (non-dropped) events
                if filtered_events:
                    # Add events to the window (inverted Y-axis)
                    window[[event.y for event in filtered_events], [event.x for event in filtered_events]] = 255
                    numevs += len(filtered_events)  # Increment event count for processed events

# Start the event-fetching thread
event_thread = threading.Thread(target=fetch_events)
event_thread.daemon = True  # Ensures the thread stops when the main program exits
event_thread.start()

# Main loop for visualization
while True:
    current_time = time.time()
    with events_lock:
        # Update the plot if enough time has passed
        if current_time - last_update_time > update_interval:
            if numevs > 0:  # Only update the display if there are events
                cv2.imshow('DVS Events', window)  # Display the window using OpenCV
                cv2.waitKey(1)  # Wait for 1 ms to allow the window to update

                # Clear the window and reset the event counter
                window.fill(0)  # Efficiently reset window to zeros
                numevs = 0
            last_update_time = current_time  # Update the last refresh timestamp