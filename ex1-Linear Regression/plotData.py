import pandas as pd
import matplotlib.pyplot as plot

def plotData(data, theta=[]):
    print('data.head and describe data as follows')
    print(data.head())
    print(data.describe())

    # data.plot(kind='scatter', x='population', y='profit')
    plot.scatter(data.population, data.profit, label='training data')
    if theta.any():
        plot.plot(data.population, theta[0,0] + data.population * theta[0,1], label='Linear regression')

    #plt.legend(loc=2)
    plot.show()