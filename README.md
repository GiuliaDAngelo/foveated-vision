# Foveated Vision

This library uses [Sinabs](https://sinabs.readthedocs.io/en/v2.0.0/) and Leaky Integrate-and-Fire (LIF) 
neurons to simulate a foveated or uniform retina. 
You'll find tutorial on how to use LIF from sinabs library and scripts to learn how to use the [Speck](https://www.synsense.ai/products/speck-2/) device. 

The goal is to create receptive fields that mimic the human retina, where the size
of the receptive fields increases from the fovea (center) to the periphery. 
The library can be used to simulate various
visual processing tasks with biologically inspired models of vision.

## Tutorials

- [tutorial_single_neuron](https://github.com/GiuliaDAngelo/foveated-vision/blob/main/tutorials/tutorial_single_neuron.py)
- [tutorial_retina](https://github.com/GiuliaDAngelo/foveated-vision/blob/main/tutorials/tutorialretina.py)
- [tutorial_retina](https://github.com/GiuliaDAngelo/foveated-vision/blob/main/tutorials/tutorialretina.py)
- [tutorial_create_eccentric_retina](https://github.com/GiuliaDAngelo/foveated-vision/blob/main/tutorials/tutorial_create_eccentric_retina.py)


## Let's start! 

- ### [tutorial_single_neuron](https://github.com/GiuliaDAngelo/foveated-vision/blob/main/tutorials/tutorial_single_neuron.py)

You will learn how to create a single LIF, inject it with current and plot the membrane potential and spikes.

- ### [tutorial_retina](https://github.com/GiuliaDAngelo/foveated-vision/blob/main/tutorials/tutorialretina.py)

You will learn how to create a retina, 128x128 receptive fields and plot the raster plot.

![firsttutorials](images/tutorialsingleneuronretina.png)

- ### [tutorial_create_eccentric_retina](https://github.com/GiuliaDAngelo/foveated-vision/blob/main/tutorials/tutorial_create_eccentric_retina.py)

You will create a retina with eccentric receptive fields, 128x128 receptive fields and plot the raster plot.
The eccentric receptive fields are created by using the `eccentricity` parameters in the `create_eccentric_RFs()` function.
The parameters are in helpers/config.py file.
In this script you will also plot the eccentric structure and the Look Up Table (LUT) of the receptive fields.
The LUT will be created so that it'll speed up the process of finding the receptive field of a given neuron.

- ### [tutorial_eccentric_retina](https://github.com/GiuliaDAngelo/foveated-vision/blob/main/tutorials/tutorial_eccentric_retina.py)

In this tutorial you will create the eccentric retina and visualise the output spikes of the retina.

![secondtutorials](images/tutorialeccentricretina.gif)

- ### [tutorialSpeckVisualiser](https://github.com/GiuliaDAngelo/foveated-vision/blob/main/tutorials/tutorialSpeckVisualiser.py)

Time to connect your Speck to your computer and visualise the output of the sensor!

![visualiser](https://github.com/GiuliaDAngelo/foveated-vision/blob/main/images/Speckvisualiser.gif)


## Installation
Have a look at the [requirements.txt](https://github.com/GiuliaDAngelo/foveated-vision/blob/main/requirements.txt) file to see the dependencies.

### Features:
- **Foveated Receptive Fields**: These fields allow for a more detailed visual representation at the center (fovea), while progressively coarser representation occurs as you move towards the periphery. This mimics the behavior of the human visual system.
- **Leaky Integrate-and-Fire Neurons**: These neurons are implemented using the Sinabs library's LIF model, which captures the spiking dynamics essential for neuromorphic computing.
  
### Components:
- **Retina Layer**: This module creates a foveated retina where the receptive field size changes according to the distance from the center. You can adjust the density of neurons and their activation patterns.
- **Spiking Neural Network (SNN) Processing**: The input from the retina is processed by a spiking neural network using LIF neurons. The Sinabs library provides the underlying infrastructure for creating and training these networks.


