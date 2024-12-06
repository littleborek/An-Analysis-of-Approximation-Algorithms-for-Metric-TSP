import numpy as np


coordinates = np.load("dataset_1_coordinates.npy")
distance_matrix = np.load("dataset_1_distance_matrix.npy")


print("Coordinates Shape:", coordinates.shape)
print("Distance Matrix Shape:", distance_matrix.shape)
print("First 5 Coordinates:\n", coordinates[:5])
