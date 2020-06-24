# localsearch_tsp
Implementation and testing of Local Search algorithms for the Travelling Salesman Problem (TSP)

---
## Table of contents

- [Initialization](#initialization)
- [Point Connection](#point-connection)
- [Nearest Neighbour](#nearest-neighbour)
- [Randomized Nearest Neighbour](#randomized-nearest-neighbour)
- [PC with 2opt](#pc-with-2opt)
- [NN with 2opt](#nn-with-2opt)
- [RNN with 2opt](#rnn-with-2opt)
- [Results](#results)

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
[![POINT CONNECTION](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/PC.png)]()

---

## Nearest Neighbour
> Starting from the first node it connects it to the nearest one and so on.
```python
NN = NearestNeighbour()
t.solve(NN)
t.plotSolution('NearestNeighbour')
print('NN cost: ', t.getCost('NearestNeighbour'))
```
[![NEAREST NEIGHBOUR](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/NN.png)]()

---

## Randomized Nearest Neighbour
> Starting from a first random node it connects it to the nearest one and so on.
```python
RNN = RandomizedNearestNeighbour()
t.solve(RNN)
t.plotSolution('RandomizedNearestNeighbour')
print('RNN cost: ', t.getCost('RandomizedNearestNeighbour'))
```
[![RANDOMIZED NEAREST NEIGHBOUR](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/RNN.png)]()

---

## PC with 2opt
> Starting from a first solution (PC) it optimizes it applying the 2-opt algorithm.
```python
initial_path = t.path
initial_cost = t.computeCost(initial_path)
G2Opt = TwoOpt(initial_path=initial_path, initial_cost=initial_cost)
t.solve(G2Opt)
t.plotSolution('TwoOpt')
print('G2Opt cost: ', t.getCost('TwoOpt'))
```
[![PC + 2OPT](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/PC2Opt.png)]()

---

## NN with 2opt
> Starting from a first solution (NN) it optimizes it applying the 2-opt algorithm.
```python
NN2Opt = TwoOptNN()
t.solve(NN2Opt)
t.plotSolution('TwoOptNN')
print('NN2Opt cost: ', t.getCost('TwoOptNN'))
```
[![NN + 2OPT](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/NN2Opt.png)]()

---

## RNN with 2opt
> Starting from a first solution (RNN) it optimizes it applying the 2-opt algorithm.
```python
RNN2Opt = TwoOptRNN()
t.solve(RNN2Opt)
t.plotSolution('TwoOptRNN')
print('RNN2Opt cost: ', t.getCost('TwoOptRNN'))
```
[![RNN + 2OPT](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/RNN2Opt.png)]()

---

## Results
> The results from the previous algorithms.
```python
t.getResults()
```
| Algorithm |    Cost   |   Time  |
|:---------:|:---------:|:-------:|
| PC        | 844745.58 |   0.001 |
| NN        | 102594.36 |   0.016 |
| RNN       |  98205.10 |   0.020 |
| PC+2Opt   |  84957.10 | 443.285 |
| NN+2Opt   |  86474.13 | 131.252 |
| RNN+2Opt  |  85704.51 | 127.434 |

---
