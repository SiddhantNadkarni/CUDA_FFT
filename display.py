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
plt.savefig('input_cuFFT_Sin_Wave.png')

outputchart = pd.read_table(outputdatafile)
outputchart.plot()
plt.title('Output data (FFT)')
plt.savefig('output_cuFFT_abs.png')
plt.show()