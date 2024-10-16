'''
Giulia D'Angelo, giulia.dangelo@fel.cvut.cz

This script simulate the eccentric representation of the retina using log-polar mapping.
The code does take isnpiration from the following article:

Chessa, M., Maiello, G., Bex, P. J., & Solari, F. (2016). A space-variant model for motion interpretation across the visual field. Journal of vision, 16(2), 12-12.
'''


import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from config import *


if __name__ == "__main__":
    [neurons, mask, ax] = create_eccentric_RFs()
    plot_RFs(neurons, mask, ax)
    plot_mask(mask, width, height)
