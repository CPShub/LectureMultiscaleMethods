# Import required modules
import numpy as np                        # numerics
import scipy as sp
from scipy import linalg
import matplotlib.pyplot as plt           # plot environment
from scipy.integrate import odeint        # numerical ODE solver
import ipywidgets as widgets              # interactive widgets
from ipywidgets import interactive, interact
from IPython.display import Javascript    # For external cell execution
import sys
from matplotlib import interactive as mat_int		  # Interactive plots
interactive(True)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import warnings 						   # Supress "NaN"-Warning at surf-plot
from sympy import I, Matrix, symbols
from sympy.physics.quantum import TensorProduct


# Add subfolders to working directory
sys.path.insert(0,"code")
sys.path.insert(0,"./code/03_kinetics")

# Function for interactive input with widgets
def inp(**kargs):
    return kargs
	
# Print options for numpy
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)}) # decimal places numpy output
