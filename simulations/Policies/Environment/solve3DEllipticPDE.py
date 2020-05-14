import sys
import os
import numpy as nu
from matplotlib import pyplot as pl
import pandas as pd
from matplotlib import cm
from mpl_toolkits import mplot3d


args = sys.argv
planeDatas = []
for filePath in args[1:]:
    if filePath == 'xSamplePoints.csv':
        xSamplesDF = pd.read_csv(filePath, header=None)
    elif filePath == 'ySamplePoints.csv':
        ySamplesDF = pd.read_csv(filePath, header=None)
    elif filePath == 'zSamplePoints.csv':
        zSamplesDF = pd.read_csv(filePath, header=None)
    else:
        # zIndex = int(filePath.split('(')[1].split(')')[0])
        planeDatas.append(pd.read_csv(filePath, header=None))

maxData = 0
minData = float('inf')
for data in planeDatas:
    maxData = max(max(data.max()), maxData)
    minData = min(min(data.min()), minData)
normalize = lambda x, max, min: (x-min)/(max-min)

flattenedXs = []
flattenedYs = []
flattenedZs = []
colorData = []
sizeData = []
fig = pl.figure()
ax = pl.axes(projection='3d')
for xIndex, xValue in enumerate(xSamplesDF.values):
    for yIndex, yValue in enumerate(ySamplesDF.values):
        for zIndex, zValue in enumerate(zSamplesDF.values):
            data = planeDatas[zIndex].iat[xIndex, yIndex]
            flattenedXs.append(xValue)
            flattenedYs.append(yValue)
            flattenedZs.append(zValue)
            colorData.append(data)
            sizeData.append(nu.sqrt(normalize(data, maxData, minData)) * 5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.scatter(flattenedXs, flattenedYs, flattenedZs, s=sizeData, c=colorData, cmap='viridis')
pl.show()

shouldDeleteData = input('Should delete data?[y/n]')
if shouldDeleteData == 'y':
    for filePath in args[1:]:
        os.remove(filePath)
