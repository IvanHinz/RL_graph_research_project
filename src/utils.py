import numpy as np

def moving_average(data, window_size=50):
    moving_averages = []
    for i in range(len(data) - window_size + 1):
        # extract the data for the next window size elements
        window = data[i: i + window_size]
        # compute the average
        window_average = np.mean(window)
        moving_averages.append(window_average)

    # return moving averages lists
    return np.array(moving_averages)