import numpy as np
from scipy.spatial import distance_matrix

# Function to generate random datasets with city coordinates and distance matrices
def generate_multiple_datasets(num_datasets, num_cities, seed=42):
    np.random.seed(seed)
    datasets = []
    for i in range(num_datasets):
        coordinates = np.random.rand(num_cities, 2)  # Random coordinates in [0, 1] range
        dist_matrix = distance_matrix(coordinates, coordinates)
        datasets.append((coordinates, dist_matrix))
    return datasets

# Generate 10 datasets for 1000 cities each
num_datasets = 10
num_cities = 1000
datasets = generate_multiple_datasets(num_datasets, num_cities)

# Example to access a dataset
# Each dataset is a tuple: (coordinates, distance_matrix)
# For example:
dataset_1 = datasets[0]
coordinates_1 = dataset_1[0]  # Coordinates of the first dataset
distance_matrix_1 = dataset_1[1]  # Distance matrix of the first dataset

# To save datasets as individual files, you can export them as needed:
for idx, (coordinates, dist_matrix) in enumerate(datasets):
    np.save(f"dataset_{idx + 1}_coordinates.npy", coordinates)
    np.save(f"dataset_{idx + 1}_distance_matrix.npy", dist_matrix)
