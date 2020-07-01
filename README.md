# localsearch_tsp
Implementation and testing of Local Search algorithms for the Travelling Salesman Problem (TSP)

---
## Table of contents

- [Initialization](#initialization)
- [Nearest Neighbour](#nearest-neighbour)
- [Repeated Nearest Neighbour](#repeated-nearest-neighbour)
- [Nearest Neighbour with 2opt](#nearest-neighbour-with-2opt)
- [Repeated Nearest Neighbour with 2opt](#repeated-nearest-neighbour-with-2opt)
- [Nearest Neighbour with 2opt and DLB](#nearest-neighbour-with-2opt-and-dlb)
- [Repeated Nearest Neighbour with 2opt and DLB](#repeated-nearest-neighbour-with-2opt-and-dlb)
- [Nearest Neighbour with 3opt](#nearest-neighbour-with-3opt)
- [Repeated Nearest Neighbour with 3opt](#repeated-nearest-neighbour-with-3opt)
- [Nearest Neighbour with 3opt and DLB](#nearest-neighbour-with-3opt-and-dlb)
- [Repeated Nearest Neighbour with 3opt and DLB](#repeated-nearest-neighbour-with-3opt-and-dlb)
- [Results](#results)

---

## Initialization
> How to initialize the environment. Note that you can either import a file using the TSP Loader class or generate how many nodes you want at random via the Generator class. 
```python
# Importing of the needed libraries
import localsearch.tsp as tsp
import localsearch.solvers as slv

# Instance of Loader class used to load the .tsp file into a suitable data-structure
data = tsp.Loader(r'..\\data\\xqf131.tsp')

# Instance of Generator class used to randomly generate a given number of nodes
# data = tsp.Generator(40)

# Instance of TSP class to which we pass the nodes of the .tsp file
t = tsp.TSP(data.nodes)

# Plotting the initial data
t.plotData()
```
[![INITIAL DATA](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/xqf131.png)]()

---

## Nearest Neighbour
> Starting from the first node it connects it to the nearest one and so on.
```python
NN = slv.NN()
t.solve(NN)
t.plotSolution('NN')
```
[![NEAREST NEIGHBOUR](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/xqf131_nn.png)]()

---

## Repeated Nearest Neighbour
> It applies the Nearest Neighbour algorithm to all the nodes (considering each one of them as first node) and then it keeps the best (the cheapest one).
```python
RepNN = slv.RepNN()
t.solve(RepNN)
t.plotSolution('RepNN')
```
[![REPEATED NEAREST NEIGHBOUR](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/xqf131_rep_nn.png)]()

---

## Nearest Neighbour with 2opt
> Starting from a first solution (NN) it optimizes it applying the 2-opt algorithm.
```python
NN2Opt = slv.NN2Opt()
t.solve(NN2Opt)
t.plotSolution('NN2Opt')
```
[![NN + 2OPT](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/xqf131_nn_2opt.png)]()

---

## Repeated Nearest Neighbour with 2opt
> Starting from a first solution (RepNN) it optimizes it applying the 2-opt algorithm.
```python
RepNN2Opt = slv.RepNN2Opt()
t.solve(RepNN2Opt)
t.plotSolution('RepNN2Opt')
```
[![RepNN + 2OPT](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/xqf131_rep_nn_2opt.png)]()

---

## Nearest Neighbour with 2opt and DLB
> Starting from a first solution (NN) it optimizes it applying the 2-opt algorithm and DLB.
```python
NN2OptDLB = slv.NN2OptDLB()
t.solve(NN2OptDLB)
t.plotSolution('NN2OptDLB')
```
[![NN + 2OPT + DLB](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/xqf131_nn_2opt_dlb.png)]()

---

## Repeated Nearest Neighbour with 2opt and DLB
> Starting from a first solution (RepNN) it optimizes it applying the 2-opt algorithm and DLB.
```python
RepNN2OptDLB = slv.RepNN2OptDLB()
t.solve(RepNN2OptDLB)
t.plotSolution('RepNN2OptDLB')
```
[![RepNN + 2OPT + DLB](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/xqf131_rep_nn_2opt_dlb.png)]()

---

## Nearest Neighbour with 3opt
> Starting from a first solution (NN) it optimizes it applying the 3-opt algorithm.
```python
NN3Opt = slv.NN3Opt()
t.solve(NN3Opt)
t.plotSolution('NN3Opt')
```
[![NN + 3OPT](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/xqf131_nn_3opt.png)]()

---

## Repeated Nearest Neighbour with 3opt
> Starting from a first solution (RepNN) it optimizes it applying the 3-opt algorithm.
```python
RepNN3Opt = slv.RepNN3Opt()
t.solve(RepNN3Opt)
t.plotSolution('RepNN3Opt')
```
[![RepNN + 3OPT](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/xqf131_rep_nn_3opt.png)]()

---

## Nearest Neighbour with 3opt and DLB
> Starting from a first solution (NN) it optimizes it applying the 3-opt algorithm and DLB.
```python
NN3OptDLB = slv.NN3OptDLB()
t.solve(NN3OptDLB)
t.plotSolution('NN3OptDLB')
```
[![NN + 3OPT + DLB](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/xqf131_nn_3opt_dlb.png)]()

---

## Repeated Nearest Neighbour with 3opt and DLB
> Starting from a first solution (RepNN) it optimizes it applying the 3-opt algorithm and DLB.
```python
RepNN3OptDLB = slv.RepNN3OptDLB()
t.solve(RepNN3OptDLB)
t.plotSolution('RepNN3OptDLB')
```
[![RepNN + 3OPT + DLB](https://raw.githubusercontent.com/L4plac3/localsearch_tsp/master/images/xqf131_rep_nn_3opt_dlb.png)]()

---

## Results
> The results from the previous algorithms.
```python
t.printResults()
```
| METHOD | COST | TIME |
|-|-|-|
| NN | 709.52 | 0.001 |
| RepNN | 628.72 | 0.162 |
| NN2Opt | 611.46 | 0.565 |
| RepNN2Opt | 588.99 | 0.319 |
| NN2OptDLB | 617.12 | 0.064 |
| RepNN2OptDLB | 588.0 | 0.032 |
| NN3Opt | 596.09 | 76.672 |
| RepNN3Opt | 582.67 | 46.277 |
| NN3OptDLB | 592.81 | 23.899 |
| RepNN3OptDLB | 583.95 | 18.987 |

---
