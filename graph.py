import numpy as np
import networkx as nx
import random

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def improved_christofides_algorithm(distance_matrix):
    num_nodes = distance_matrix.shape[0]

    # Calculate average weight of edges in the complete graph
    total_weight = np.sum(distance_matrix) / 2  # Since it's a symmetric distance matrix
    avg_weight = total_weight / (num_nodes * (num_nodes - 1) / 2)

    epsilon = adjust_epsilon(num_nodes, avg_weight)

    # Step 1: Create a complete graph
    G = nx.Graph()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            G.add_edge(i, j, weight=distance_matrix[i][j])

    # Step 2: Compute a random spanning tree based on LP relaxation
    T = get_epsilon_mst(G, epsilon)

    # Step 3: Find nodes with odd degree in MST
    odd_degree_nodes = [v for v, degree in T.degree() if degree % 2 == 1]

    # Step 4: Create a subgraph induced by odd degree nodes and compute Minimum Weight Perfect Matching
    subgraph = G.subgraph(odd_degree_nodes)
    matching = nx.algorithms.matching.min_weight_matching(subgraph)

    # Step 5: Combine MST and matching edges to form a multigraph
    multigraph = nx.MultiGraph()
    multigraph.add_edges_from(T.edges(data=True))
    multigraph.add_edges_from(matching)

    # Step 6: Find an Eulerian circuit in the multigraph
    euler_circuit = list(nx.eulerian_circuit(multigraph))

    # Step 7: Make the circuit Hamiltonian by shortcutting repeated vertices
    visited = set()
    hamiltonian_path = []
    for u, v in euler_circuit:
        if u not in visited:
            visited.add(u)
            hamiltonian_path.append(u)
    hamiltonian_path.append(hamiltonian_path[0])  # Make it a cycle

    # Calculate the total cost
    total_cost = 0
    for i in range(len(hamiltonian_path) - 1):
        u = hamiltonian_path[i]
        v = hamiltonian_path[i + 1]
        total_cost += distance_matrix[u][v]

    return hamiltonian_path, total_cost


def adjust_epsilon(num_nodes, avg_weight):
    base_epsilon = 1 / (num_nodes ** 0.5)
    weight_factor = min(1, 1 / (avg_weight + 1))  # Limit impact of weight to between 0 and 1
    return max(1e-6, min(5e-3, base_epsilon * weight_factor))


def get_epsilon_mst(G, epsilon):
    edges = list(G.edges(data=True))
    random.shuffle(edges)

    mst = nx.Graph()
    mst.add_nodes_from(G.nodes)

    components = {node: node for node in G.nodes}

    def find(node):
        while components[node] != node:
            node = components[node]
        return node

    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            components[root_v] = root_u

    for u, v, data in sorted(edges, key=lambda x: x[2]['weight'] * (1 + random.uniform(-epsilon, epsilon))):
        if find(u) != find(v):
            mst.add_edge(u, v, weight=data['weight'])
            union(u, v)
        if len(mst.edges) == len(G.nodes) - 1:
            break

    return mst


def plot_tsp_route(coordinates, route, dataset_idx):

    plt.figure(figsize=(8, 6))
    x_coords, y_coords = coordinates[:, 0], coordinates[:, 1]
    plt.scatter(x_coords, y_coords, c='blue', label='Cities')

    # Draw the route
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        plt.plot([x_coords[u], x_coords[v]], [y_coords[u], y_coords[v]], 'r-')

    plt.title(f"Improved TSP Route for Dataset {dataset_idx}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()

    # Save the plot as an SVG file (no quality loss)
    plt.savefig(f"dataset_{dataset_idx}_improved_random.svg", format="svg")

    # Display the plot
    plt.show()

a=1
# Main execution for datasets 1 to 10
while a==1:
    for i in [7, ]:

    # Load the distance matrix and coordinates
        distance_matrix = np.load(f'dataset_{i}_distance_matrix.npy')
        coordinates = np.load(f'dataset_{i}_coordinates.npy')

    # Run the improved Christofides algorithm
        tour, cost = improved_christofides_algorithm(distance_matrix)

    # Print the results
        print(f'Dataset {i}: Total Cost = {cost}')
        if cost <= 26.114943100717117:
            a=0

    # Plot the TSP route


plot_tsp_route(coordinates, tour, i)