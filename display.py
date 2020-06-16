import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


inputdatafile = "input_file.dat"
outputdatafile = "output_file.dat"

inputchart = pd.read_table(inputdatafile)
inputchart.plot()
plt.title('Input data')
plt.xlabel('time')
plt.ylabel('Amplitude')

outputchart = pd.read_table(outputdatafile)
outputchart.plot()
plt.title('Output data (FFT)')

plt.show()