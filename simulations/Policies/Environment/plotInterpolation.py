import sys
import os
import numpy as nu
from matplotlib import pyplot as pl
import pandas as pd
from matplotlib import cm
from mpl_toolkits import mplot3d


fig = pl.figure()
for path in sys.argv[1:]:
    data = pd.read_csv(path, index_col=0, header=0)
    xs = data.index.values[:, nu.newaxis]
    ys = nu.array([ float(name) for name in data.columns.values])[nu.newaxis, :]
    values = data.values
    ax = pl.axes(projection='3d')
    ax.plot_surface(xs, ys, values, cmap='viridis')
    pl.show()
