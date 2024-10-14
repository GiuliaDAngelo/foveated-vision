import os
os.environ['ApplePersistenceIgnoreState'] = 'YES'
import torch
import sinabs.layers as sl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from helpers import *


class RFs:
    def __init__(self, tau_mem):
        self.neuron = sl.LIF(tau_mem=tau_mem)
        self.vmem = []

