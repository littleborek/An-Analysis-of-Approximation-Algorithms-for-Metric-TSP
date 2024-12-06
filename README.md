# Approximation Algorithms for Metric Traveling Salesman Problem (TSP) - Python

This repository contains implementations of three algorithms for solving the Metric Traveling Salesman Problem (TSP):

1. **Greedy Algorithm**:  
   A simple nearest-neighbor approach where the algorithm iteratively selects the nearest unvisited city.

2. **Christofides Algorithm**:  
   An approximation algorithm for Metric TSP, which uses a Minimum Spanning Tree (MST), a perfect matching for odd-degree vertices, and an Eulerian tour.

3. **Slightly Improved Christofides Algorithm**:  
   A randomized version of Christofides' algorithm that introduces minor refinements (such as epsilon adjustments) to improve the approximation ratio.

The algorithms are compared in terms of total cost and execution time, with visual representations of the computed tours for each dataset.

## Results

### Metric Evaluation
The total cost of the computed tours for each algorithm is compared.

### Spatial Comparison
The computed routes are visualized and compared on a 2D plane for each dataset.

![4 fin](https://github.com/user-attachments/assets/98b38259-b755-4496-8092-0d4a66bf0a03)

## Getting Started

To run the algorithms and reproduce the results:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-repository-name/approximation-algorithms-tsp.git
    cd approximation-algorithms-tsp
    ```

2. Install the required libraries:

    ```bash
    pip install numpy networkx matplotlib scipy
    ```

3. Run the script to generate datasets and evaluate the algorithms:

    ```bash
    python compare_algorithms.py
    ```

### Dataset Generation
The `dataset_generator.py` script will automatically generate datasets consisting of random cities. Each dataset will have 1000 cities, and the corresponding distance matrix will be computed based on Euclidean distances between cities.

### Graphical Output
The code generates graphical outputs (in `.svg` format) for each dataset, allowing you to visualize the computed routes. The graphs will show the cities' coordinates in a 2D plane and the path taken by each algorithm to visit all cities.

## Algorithms Overview

1. **Greedy Algorithm**:  
   A simple nearest-neighbor approach where the algorithm iteratively selects the nearest unvisited city.

2. **Christofides Algorithm**:  
   A approximation algorithm for Metric TSP, which uses a Minimum Spanning Tree (MST), a perfect matching for odd-degree vertices, and an Eulerian tour.

3. **Slightly Improved Christofides Algorithm**:  
   A randomized version of Christofides' algorithm that introduces minor refinements (such as epsilon adjustments) to improve the approximation ratio.
