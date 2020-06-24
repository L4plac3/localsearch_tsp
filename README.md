# localsearch_tsp
Implementation and testing of Local Search algorithms for the Travelling Salesman Problem (TSP)

---
## Table of contents

- [Initialization](#initialization)
- [Point Connection](#point-connection)
- [Nearest Neighbour](#nearest-neighbour)
- [Randomized Nearest Neighbour Solver](#randomized-nearest-neighbour)
- [Generic 2-opt Solver](#generic-2opt)
- [Nearest Neighbour with 2-opt Solver](#nearest-neighbour-2opt)
- [Randomized Nearest Neighbour with 2-opt Solver](#randomized-nearest-neighbour-2opt)

---

## Initialization
> How to initialize the environment.
```python
# Importing of the needed libraries
from localsearch import *

# TSPloader class used to load the .tsp file into a suitable data-structure
loader = TSPloader(r'..\\data\\uy734.tsp')

# TSP class to which we pass the nodes of the .tsp file
t = TSP(loader.nodes)

# Plotting the initial data
t.plotData()
```
[![INITIAL DATA](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/initialdata.png)]()

---

## Point Connection
> It connects the nodes as they appear in the input file.
```python
PC = PointConnection()
t.solve(PC)
t.plotSolution('PointConnection')
print('PC cost: ', t.getCost('PointConnection'))
```
[![INITIAL DATA](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/PC.png)]()

---

## Nearest Neighbour
> Starting from the first node it connects it to the nearest one and so on.
```python
NN = NearestNeighbour()
t.solve(NN)
t.plotSolution('NearestNeighbour')
print('NN cost: ', t.getCost('NearestNeighbour'))
```
[![INITIAL DATA](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/NN.png)]()

---

## Randomized Nearest Neighbour
> Starting from a first random node it connects it to the nearest one and so on.
```python
RNN = RandomizedNearestNeighbour()
t.solve(RNN)
t.plotSolution('RandomizedNearestNeighbour')
print('RNN cost: ', t.getCost('RandomizedNearestNeighbour'))
```
[![INITIAL DATA](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/RNN.png)]()

---

## Generic 2opt
> Starting from a first solution (PC) it optimizes it applying the 2-opt algorithm.
```python
initial_path = t.path
initial_cost = t.computeCost(initial_path)
G2Opt = TwoOpt(initial_path=initial_path, initial_cost=initial_cost)
t.solve(G2Opt)
t.plotSolution('TwoOpt')
print('G2Opt cost: ', t.getCost('TwoOpt'))
```
[![INITIAL DATA](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/G2Opt.png)]()

---

## Nearest Neighbour 2opt
> Starting from a first solution (NN) it optimizes it applying the 2-opt algorithm.
```python
NN2Opt = TwoOptNN()
t.solve(NN2Opt)
t.plotSolution('TwoOptNN')
print('NN2Opt cost: ', t.getCost('TwoOptNN'))
```
[![INITIAL DATA](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/NN2Opt.png)]()

---

## Randomized Nearest Neighbour 2opt
> Starting from a first solution (RNN) it optimizes it applying the 2-opt algorithm.
```python
RNN2Opt = TwoOptRNN()
t.solve(RNN2Opt)
t.plotSolution('TwoOptRNN')
print('RNN2Opt cost: ', t.getCost('TwoOptRNN'))
```
[![INITIAL DATA](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/RNN2Opt.png)]()

---

## Results
> The results from the previous algorithms.
```python
t.getResults()
```
| Algorithm |    Cost   |  Time  |
|:---------:|:---------:|:------:|
| PC        | 844745.58 |  0.001 |
| NN        | 102594.36 |  0.020 |
| RNN       | 101227.06 |  0.022 |
| PC+2OPt   |  88096.62 | 12.314 |
| NN+2Opt   |  86474.13 |  6.124 |
| RNN+2Opt  |  85704.51 | 10.633 |

---
