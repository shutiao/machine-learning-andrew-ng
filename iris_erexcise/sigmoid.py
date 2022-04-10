import numpy as np

# %SIGMOID Compute sigmoid function
def sigmoid(z):
    # Compute the sigmoid of each value of z (z can be a matrix vector or scalar).
    # Return g = zeros(size(z));
    return 1 / ( 1 + np.exp(-z))

def sanity_check():
    import matplotlib.pyplot as plot 
    nums = np.arange(-10, 10)
    plot.plot(nums, sigmoid(nums))
    plot.show()

if __name__ == "__main__":
    sanity_check()