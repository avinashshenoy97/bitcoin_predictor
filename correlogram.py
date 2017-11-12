from pandas import Series
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot

from init import *


ipdata = init()
# Plot autocorrelation plot
plot_acf(ipdata['next'])
pyplot.show()

# First 70 have significant lag valuess