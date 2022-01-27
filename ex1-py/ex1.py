# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import logging
import plotData
import computeCost

# Machine Learning Online Class - Exercise 1: Linear Regression
""" %     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m """

""" x refers to the population size in 10,000s
% y refers to the profit in $10,000s """

print('# ======================= Part 2: Plotting =======================')

path = 'ex1data1.txt'
data = pd.read_csv(path, names = ['population', 'profit'])
plotData.plot_data(data)

