# Wenzhe Zheng
# 3/27/2021
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the  dataset
data_set = scipy.io.loadmat("./AllSamples.mat")

samples = data_set["AllSamples"]

s_lot = []

# Convert the dataset into  Dataframe
points = pd.DataFrame(samples)

# draw the distribution of the given data
plt.figure(figsize=(7, 7))
plt.scatter(points[0], points[1], color='g', alpha=0.7, edgecolor='b')
plt.title("The distribution of the given samples")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

colors = ['r', 'c', 'cadetblue', 'blue', 'teal', 'g', 'aqua', 'lightskyblue', 'blueviolet', 'cyan', 'steelblue', 'slateblue']

# Assignment function used to assign each point to the nearest centroid
def assignment(points, centroids):
    global colors
    """
    :param points: the sample dataframe
    :param centroids: the current centroids
    calculate the distance from points to each centroids and the select the closest one
    :return: the new dataframe
    """
    for i in centroids.keys():
        points['distance_from_{}'.format(i)] = (
            np.sqrt(
                (points[0] - centroids[i][0]) ** 2 +
                (points[1] - centroids[i][1]) ** 2
            )
        )
    # Calculate the distance of each point from every centroids and assign it to the closest one
    centroid_distance_closest = ['distance_from_{}'.format(i) for i in centroids.keys()]
    points['closest'] = points.loc[:, centroid_distance_closest].idxmin(axis=1)
    points['closest'] = points['closest'].map(lambda x: int(x.lstrip('distance _from_')))
    points['color'] = points['closest'].map(lambda x: colors[x])
    return points


def update(centroids):
    """
    update the centroids
    :return:
    """
    for i in centroids.keys():
        centroids[i][0] = np.mean(points[points['closest'] == i][0])
        centroids[i][1] = np.mean(points[points['closest'] == i][1])
    return centroids

for k in range(2, 11):
    n = len(points[0])
    # Strategy 1: randomly pick the initial centers from the given samples.
    centroids = {}
    for i in range(k):
        index = np.random.randint(0, n)
        centroids[i+1] = [points[0][index], points[1][index]]

    # Print the initial Centroids
    print("the initial centroids are")
    print(centroids)

    # First assignment step performed to assign the points to the calculated centroids
    points = assignment(points, centroids)
    # First update step performed to update the first set of Centroids
    centroids = update(centroids)

    # while loop to perform the above mentioned steps in a iterative fashion until the centroids converge.
    while True:
        points = assignment(points, centroids)
        old_centroids = points['closest'].copy(deep=True)
        centroids = update(centroids)

        # break when the centroids don't change
        if old_centroids.equals(points['closest']):
            break
    # draw the result
    print("Centroids", centroids)
    fig = plt.figure(figsize=(7, 7))
    plt.scatter(points[0], points[1], color=points['color'], alpha=0.7, edgecolor='b')

    for i in centroids.keys():
        plt.scatter(*centroids[i], color='red')

    plt.xlim(0, 10)
    plt.ylim(-1, 10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('The distribution with ' + str(k) + ' clusters')
    plt.show()

    # Calculation of the objective function
    sum_dist = 0
    for i in centroids.keys():
        sum_dist = sum_dist + np.sum((points[points['closest'] == i][0] - centroids[i][0]) ** 2) + np.sum(
            (points[points['closest'] == i][1] - centroids[i][1]) ** 2)

    s_lot.append(sum_dist)

# Plot the Objective function graph
k_dist = [2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.figure(figsize=(7, 7))
plt.plot(k_dist, s_lot)
plt.scatter(k_dist, s_lot)
plt.title('Strategy 1 - Objective Function vs Number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Objective Function Value')
plt.show()
