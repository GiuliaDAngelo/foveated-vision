# Foveated Vision

This library uses [Sinabs](https://synsense-sinabs.readthedocs-hosted.com/en/latest/) and Leaky Integrate-and-Fire (LIF) neurons to simulate a foveated retina. The goal is to create receptive fields that mimic the human retina, where the size of the receptive fields increases from the fovea (center) to the periphery. The library can be used to simulate various visual processing tasks with biologically inspired models of vision.

### Features:
- **Foveated Receptive Fields**: These fields allow for a more detailed visual representation at the center (fovea), while progressively coarser representation occurs as you move towards the periphery. This mimics the behavior of the human visual system.
- **Leaky Integrate-and-Fire Neurons**: These neurons are implemented using the Sinabs library's LIF model, which captures the spiking dynamics essential for neuromorphic computing.
  
### Components:
- **Retina Layer**: This module creates a foveated retina where the receptive field size changes according to the distance from the center. You can adjust the density of neurons and their activation patterns.
- **Spiking Neural Network (SNN) Processing**: The input from the retina is processed by a spiking neural network using LIF neurons. The Sinabs library provides the underlying infrastructure for creating and training these networks.

### Dependencies:
- [Sinabs Documentation](https://synsense-sinabs.readthedocs-hosted.com/en/latest/): A Python library for building spiking neural networks based on event-driven neuromorphic systems.

### Example Usage:

