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
- [PC with 2opt and DLB](#pc-with-2opt-and-dlb)
- [NN with 2opt and DLB](#nn-with-2opt-and-dlb)
- [RNN with 2opt and DLB](#rnn-with-2opt-and-dlb)
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

## PC with 2opt and DLB
> Starting from a first solution (PC) it optimizes it applying the 2-opt algorithm and DLB.
```python
initial_path = t.path
initial_cost = t.computeCost(initial_path)
PC2OptDLB = TwoOptDLB(initial_path=initial_path, initial_cost=initial_cost)
t.solve(PC2OptDLB)
t.plotSolution('TwoOptDLB')
print('PC2OptDLB cost: ', t.getCost('TwoOptDLB'))
```
[![PC + 2OPT + DLB](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/PC2OptDLB.png)]()

---

## NN with 2opt and DLB
> Starting from a first solution (NN) it optimizes it applying the 2-opt algorithm and DLB.
```python
NN2OptDLB = NN2OptDLB()
t.solve(NN2OptDLB)
t.plotSolution('NN2OptDLB')
print('NN2OptDLB cost: ', t.getCost('NN2OptDLB'))
```
[![NN + 2OPT + DLB](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/NN2OptDLB.png)]()

---

## RNN with 2opt and DLB
> Starting from a first solution (RNN) it optimizes it applying the 2-opt algorithm and DLB.
```python
RNN2OptDLB = RNN2OptDLB()
t.solve(RNN2OptDLB)
t.plotSolution('RNN2OptDLB')
print('RNN2OptDLB cost: ', t.getCost('RNN2OptDLB'))
```
[![RNN + 2OPT + DLB](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/RNN2OptDLB.png)]()

---

## Results
> The results from the previous algorithms.
```python
t.printResults()
```
| METHOD | COST | TIME |
|-|-|-|
| PointConnection | 844745.58 | 0.0 |
| NearestNeighbour | 102594.36 | 0.023 |
| RandomizedNearestNeighbour | 98595.23 | 0.017 |
| TwoOpt | 84657.86 | 626.933 |
| NN2Opt | 85637.62 | 230.801 |
| RNN2Opt | 85799.91 | 262.439 |
| TwoOptDLB | 90594.92 | 2.106 |
| NN2OptDLB | 87256.47 | 1.625 |
| RNN2OptDLB | 86559.63 | 1.705 |

---
