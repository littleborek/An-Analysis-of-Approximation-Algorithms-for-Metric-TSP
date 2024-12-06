import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import itertools

def christofides_algorithm(distance_matrix):
    # Step 1: Create a complete graph
    G = nx.Graph()
    num_nodes = distance_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            G.add_edge(i, j, weight=distance_matrix[i][j])

    # Step 2: Compute the Minimum Spanning Tree (MST)
    T = nx.minimum_spanning_tree(G)

    # Step 3: Find nodes with odd degree in MST
    odd_degree_nodes = [v for v, degree in T.degree() if degree % 2 == 1]

    # Step 4: Find a minimum weight perfect matching among odd degree nodes
    # Create a complete graph of the odd-degree nodes with negated weights
    odd_graph = nx.Graph()
    for u, v in itertools.combinations(odd_degree_nodes, 2):
        weight = distance_matrix[u][v]
        # Negate the weight for max_weight_matching to find the minimum weight matching
        odd_graph.add_edge(u, v, weight=-weight)

    # Find the matching
    matching = nx.algorithms.matching.max_weight_matching(odd_graph, maxcardinality=True)

    # Build the matching graph M
    M = nx.Graph()
    for u, v in matching:
        # Use the original weight (non-negated)
        M.add_edge(u, v, weight=distance_matrix[u][v])

    # Step 5: Combine MST and matching edges to form a multigraph
    multigraph = nx.MultiGraph()
    multigraph.add_edges_from(T.edges(data=True))
    multigraph.add_edges_from(M.edges(data=True))

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


# Main execution for datasets 1 to 10
with open("christofides_results.txt", "w") as f:
    for i in range(1, 11):
        # Load the distance matrix and coordinates
        distance_matrix = np.load(f'dataset_{i}_distance_matrix.npy')
        coordinates = np.load(f'dataset_{i}_coordinates.npy')

        # Run the Christofides algorithm
        tour, cost = christofides_algorithm(distance_matrix)

        # Print the results and write to the file
        result = f'Dataset {i}: Total Cost = {cost}\n'
        print(result)
        f.write(result)


print("Results saved to 'christofides_results.txt'")


