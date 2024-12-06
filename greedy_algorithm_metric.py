import numpy as np
from scipy.spatial import distance_matrix

def generate_dataset(num_cities, seed=42):

    np.random.seed(seed)
    coordinates = np.random.rand(num_cities, 2)  # Random coordinates in [0, 1] range
    dist_matrix = distance_matrix(coordinates, coordinates)
    return coordinates, dist_matrix

def greedy_tsp(distance_matrix):

    num_cities = distance_matrix.shape[0]
    visited = [False] * num_cities
    route = []

    # Start from the first city
    current_city = 0
    visited[current_city] = True
    route.append(current_city)
    total_distance = 0

    for _ in range(num_cities - 1):
        # Find the nearest unvisited city
        nearest_city = None
        min_distance = float('inf')

        for next_city in range(num_cities):
            if not visited[next_city] and distance_matrix[current_city, next_city] < min_distance:
                nearest_city = next_city
                min_distance = distance_matrix[current_city, next_city]

        # Visit the nearest city
        visited[nearest_city] = True
        route.append(nearest_city)
        total_distance += min_distance
        current_city = nearest_city

    # Return to the starting city
    total_distance += distance_matrix[current_city, route[0]]
    route.append(route[0])

    return route, total_distance

def evaluate_greedy_tsp(num_datasets, num_cities):

    results = []

    for idx in range(num_datasets):
        print(f"Generating dataset {idx + 1}...")
        coordinates, dist_matrix = generate_dataset(num_cities, seed=42 + idx)
        print(f"Evaluating greedy TSP for dataset {idx + 1}...")
        route, total_distance = greedy_tsp(dist_matrix)

        # Save dataset and result for debugging or further analysis
        np.save(f"dataset_{idx + 1}_coordinates.npy", coordinates)
        np.save(f"dataset_{idx + 1}_distance_matrix.npy", dist_matrix)
        np.save(f"result_{idx + 1}_route.npy", route)
        with open(f"result_{idx + 1}_summary.txt", "w") as f:
            f.write(f"Dataset {idx + 1}:\n")
            f.write(f"Total Distance: {total_distance:.3f}\n")  # Changed to 3 decimal places

        results.append((route, total_distance))
        print(f"Dataset {idx + 1}: Total Distance = {total_distance:.3f}")  # Changed to 3 decimal places

    return results

if __name__ == "__main__":
    NUM_DATASETS = 10
    NUM_CITIES = 1000

    all_results = evaluate_greedy_tsp(NUM_DATASETS, NUM_CITIES)

    print("\nFinal Results:")
    for idx, (_, total_distance) in enumerate(all_results):
        print(f"Dataset {idx + 1}: Total Distance = {total_distance:.3f}")  # Changed to 3 decimal places
