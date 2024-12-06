Approximation Algorithms for Metric Traveling Salesman Problem (TSP)
This repository contains the implementation of three algorithms for solving the Metric Traveling Salesman Problem (TSP):

Greedy Algorithm
Christofides Algorithm
Slightly Improved Christofides Algorithm
The algorithms are compared in terms of total cost and execution time, with visual representations of the computed tours for each dataset.

Algorithms Overview
Greedy Algorithm:
A simple nearest-neighbor approach where the algorithm iteratively selects the nearest unvisited city.

Christofides Algorithm:
A well-known approximation algorithm for Metric TSP, achieving a 
3
2
2
3
​
 -approximation ratio by constructing a Minimum Spanning Tree (MST), finding a perfect matching for odd-degree vertices, and forming an Eulerian tour.

Slightly Improved Christofides Algorithm:
A refined version of the Christofides Algorithm with randomization techniques and epsilon adjustments to improve the approximation ratio.

Requirements
Python 3.x
NumPy
NetworkX
Matplotlib
SciPy
You can install the necessary libraries using:

Copy code
pip install numpy networkx matplotlib scipy
Dataset
The experiments were conducted on 10 randomly generated datasets, each containing 1000 cities represented as points in a 2D Euclidean space.
The distance matrix for each dataset is computed based on Euclidean distances between cities.
Usage
Clone this repository:

bash
Copy code
git clone https://github.com/your-username/approximation-algorithms-tsp.git
cd approximation-algorithms-tsp
Run the Python scripts for generating datasets and comparing the algorithms:

bash
Copy code
python compare_algorithms.py
The results will be saved in .txt files for the cost evaluations and .svg files for the visual route comparisons.

Results
The algorithms were evaluated based on:

Total Cost: The sum of the distances in the computed tour.
Execution Time: The time it took to compute the solution.
Visual Comparison: A graphical representation of the computed tours for each dataset.
Future Work
Explore hybrid approaches combining Christofides' algorithm with other heuristics.
Implement parallel computing techniques to scale the algorithms to larger datasets.
Test the algorithms in real-world scenarios for logistics and robotics applications.
