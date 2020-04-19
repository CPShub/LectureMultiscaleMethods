# Import required modules
import numpy as np                        # numerics
import matplotlib.pyplot as plt           # plot environment
from scipy.integrate import odeint        # numerical ODE solver
import ipywidgets as widgets              # interactive widgets
from ipywidgets import interactive
from IPython.display import Javascript    # For external cell execution
import sys

# Add subfolders to working directory
sys.path.insert(0,"code")
sys.path.insert(0,"./code/01_michaelismenten")

# Function for interactive input with widgets
def inp(**kargs):
    return kargs