import pandas as pd
import matplotlib.pyplot as plt

def plot_data(data):
    print('data.head and describe data as follows')
    print(data.head())
    print(data.describe())
    data.plot(kind='scatter', x='population', y='profit')
    plt.show()