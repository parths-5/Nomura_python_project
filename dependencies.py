# dependencies.py
# first run this script to install necessary libraries
"""
This script contains common imports and library installations.
"""

import subprocess
# Install necessary libraries
try:
    import baostock 
except ImportError:
    # Not installed, so install it
    subprocess.check_call(['pip', 'install', 'baostock', '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple/', '--trusted-host', 'pypi.tuna.tsinghua.edu.cn'])
    

try:
    import pandas
except ImportError:
    # Not installed, so install it
    subprocess.check_call(['pip', 'install', 'pandas'])
    

try:
    import numpy
except ImportError:
    # Not installed, so install it
    subprocess.check_call(['pip', 'install', 'numpy'])

try:
    import matplotlib
except ImportError:
    # Not installed, so install it
    subprocess.check_call(['pip', 'install', 'matplotlib'])    

   

# Import libraries
import baostock as bs
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


__all__ = ['bs', 'pd', 'np', 'os','plt']