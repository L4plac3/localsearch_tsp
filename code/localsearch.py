from scipy.spatial.distance import pdist, squareform
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import random
import time





##### TSP-GENERATOR CLASS #####
class TSPgenerator():
    '''
    Basic class to generate a given number of cities at random.
    '''

    def __init__(self, number, width=900, height=600):
        '''
        TSPgenerator constructor.

        Parameters:
            - number: number of random cities.
            - width: maximum x coordinate.
            - height: maximum y coordinate.
        '''
        nodes = [[round(random.uniform(0,width)*2)/2, round(random.uniform(0,height)*2)/2] 
                for _ in range(number)]
        self.nodes = np.array(nodes)





##### TSP-LOADER CLASS #####
class TSPloader():
    '''
    Class used to load a .tsp file into a numpy array containing 
    a numpy array (x,y), where x and y are coordinates of the node.
    '''

    def __init__(self, filepath):
        '''
        Class TSPloades constructor.

        Parameters:
            - filepath : path to the .tsp file
        '''
        self.filepath = filepath
        f = open(filepath, 'r')
        nodes = []
        for line in f.readlines():
            splitted = line.strip().split()
            label = splitted[0]
            if label.isdigit():
                node = [float(splitted[1]), float(splitted[2])]
                nodes.append(node)
        self.nodes = np.array(nodes)
        f.close()





##### TSP CLASS #####
class TSP():
    '''
    TSP class, used to handle a generic instance for the TSP problem.
    '''

    def __init__(self, nodes, dist='euclidean'):
        '''
        Class TSP constructor.

        Parameters:
            - nodes : numpy matrix containing the nodes (x,y)
        '''
        self.routes = {}
        self.nodes = nodes
        self.path = list(range(len(nodes)))
        self.path.append(0)
        self.distMatFromNodes(nodes,dist)
        self.N = len(nodes)

    def distMatFromNodes(self, nodes, dist='euclidean'):
        '''
        Computes the matrix distance of the nodes in input.
        
        Parameters:
            - nodes : list of tuples (x,y), where x and y are
                    coordinates of the nodes
            - dist : the type of distance to compute 
                    (default: euclidean)
        '''
        self.dist_mat = squareform(pdist(nodes, dist))
        np.fill_diagonal(self.dist_mat, np.inf)
    
    def plotData(self):
        ''' 
        Plots the data if it has been specified.
        '''
        if self.nodes == []:
            raise Exception('No data of the instance has been loaded')
        else:
            plt.scatter(*self.nodes.T)
            plt.show()

    def solve(self, solver):
        '''
        Solves the current tsp instance using the provided solver (NN, 2-opt, ...).

        Parameters:
            - solver : object of class "Solver" 
        '''
        solver.solve(self)
        self.routes[solver.__class__.__name__] = {  'path': solver.heuristic_path,\
                                                    'cost': solver.heuristic_cost,\
                                                    'time': solver.heuristic_time}

    def plotSolution(self, route_key):
        '''
        Plots the solution found with a given solver.

        Parameters:
            - route_key : name of the solver (a.k.a. key in the routes dictionary)
        '''
        if isinstance(route_key, int):
            route_key = list(self.routes.keys())[route_key]
        route = self.routes[route_key]['path']
        plt.scatter(*self.nodes.T)
        for i in range(self.N):
            plt.plot(*self.nodes[[route[i], route[i+1]]].T, 'b')
        plt.show()

    def getResults(self):
        '''
        Gets the costs and the elapsed times of all the computed routes.
        '''
        routes = {}
        for solver, route in self.routes.items():
            routes[solver] = {  'cost': route['cost'],\
                                'time': route['time']}
        return routes
    
    def printResults(self):
        '''
        Prints the costs and the elapsed times of all the computed routes.
        '''
        print("{:<35} {:<15} {:<10}".format('METHOD','COST','TIME'))
        for solver, route in self.routes.items():
            cost, time = round(route['cost'],2), round(route['time'],3)
            print("{:<35} {:<15} {:<10}".format(solver, cost, time))

    def getCost(self, route_key):
        '''
        Get the costs of the route computed via the solver defined by route_key.

        Parameters:
            - route_key : name of the solver (a.k.a. key in the routes dictionary)
        '''
        return self.routes[route_key]['cost']

    def getPath(self, route_key):
        '''
        Get the path of the route computed via the solver defined by route_key.

        Parameters:
            - route_key : name of the solver (a.k.a. key in the routes dictionary)
        '''
        return self.routes[route_key]['path']

    def computeCost(self, path):
        '''
        Compute the cost of the given path.

        Parameters:
            - path : the path of which to compute the cost
        '''
        return sum([self.dist_mat[path[i]][path[i+1]] for i in range(len(path)-1)])





##### BASE SOLVER CLASS #####
class Solver(ABC):
    ''' Base solver class '''

    def __init__(self, initial_node=0, initial_path=None, initial_cost=None):
        ''' 
        Base solver constructor.
        
        Parameters:
            - initial_node : int 
                    The starting node for the solution.
            - initial_path : permutation of the nodes
                    Initial path to which apply the local search algorithms
        '''
        self.initial_node = initial_node
        self.initial_path = initial_path
        self.initial_cost = initial_cost

    @abstractmethod
    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        pass



##### NEAREST NEIGHBOUR SOLVER CLASS #####
class NearestNeighbour(Solver):
    ''' 
    Nearest Neighbor solver class, derived from solver. 
    '''

    def __init__(self, initial_node=0):
        ''' 
        NN constructor.
        
        Parameters:
            - initial_node : int
                    The starting node for the solution.
        '''
        super().__init__(initial_node)
    
    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via Nearest Neighbour algorithm.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        dist_mat = np.copy(tsp.dist_mat)
        current = self.initial_node
        self.heuristic_path = [current]
        self.heuristic_cost = 0
        start_time = time.time()
        while len(self.heuristic_path) < tsp.N:
            dist_mat[:, current] = np.inf
            neighbour = self.findNeighbour(current, dist_mat)
            self.heuristic_cost += dist_mat[current][neighbour]
            self.heuristic_path.append(neighbour)
            current = neighbour
        end_time = time.time()
        self.heuristic_time = end_time - start_time
        self.heuristic_cost += tsp.dist_mat[current][self.initial_node]
        self.heuristic_path.append(self.initial_node)
    
    def findNeighbour(self, node_index, dist_mat):
        return np.argmin(dist_mat[node_index])



##### RANDOMIZED NEAREST NEIGHBOUR SOLVER CLASS #####
class RandomizedNearestNeighbour(NearestNeighbour):
    ''' 
    Randomized Nearest Neighbor solver class, derived from solver. 
    The initial node is computed randomly.
    '''

    def solve(self, tsp):
        '''
        Same solve method derived from Nearest Neighbour with the only
        difference that the initial node is random.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        N = tsp.N
        self.initial_node = random.randint(0, N-1)
        super().solve(tsp)



##### 2-OPT SOLVER CLASS #####
class TwoOpt(Solver):
    '''
    Basic 2-opt algorithm.
    '''

    def __init__(self, initial_path=None, initial_cost=None, dlb=False):
        ''' 
        2-opt constructor.
        
        Parameters:
            - initial_node : int
                    The starting node for the solution.
            - initial_path : list
                    Initial path to which apply 2-opt algorithm
            - dlb : bool
                    True if Don't Look Bits is applied
        '''
        super().__init__(initial_path=initial_path, initial_cost=initial_cost)
        self.dlb = dlb

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 2-opt.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        self.heuristic_path = self.initial_path[:]
        self.heuristic_cost = self.initial_cost
        start_time = time.time()
        locally_optimal = False
        if self.dlb: self.setAllDLB(tsp.N+1, False)
        while not locally_optimal:
            locally_optimal = True
            for i in range(tsp.N-2):
                if not locally_optimal: break
                if self.dlb:
                    if self.dlb_arr[i]: continue
                    node_improved = False
                for j in range(i + 2, tsp.N - 1 if i == 0 else tsp.N):
                    gain = self.gain(i, j, tsp.dist_mat)
                    if gain > 0:
                        self.swap(i+1, j)
                        self.heuristic_cost -= gain
                        locally_optimal = False
                        if self.dlb: 
                            self.setDLB([i,i+1,j,j+1],False)
                            node_improved = True
                        break
                if self.dlb and not node_improved: self.setDLB([i],True)
        end_time = time.time()
        self.heuristic_time = end_time - start_time

    def setAllDLB(self, len, val=False):
        '''
        Function which sets the values of the DLB flag for all the nodes.
        '''
        self.dlb_arr = [val]*len
    
    def setDLB(self, nodes_list, val=False):
        '''
        Function which sets the values of the DLB flag for the nodes in nodes_list.
        '''
        for node in nodes_list:
            self.dlb_arr[node] = val
                 
    def gain(self, i, j, dist_mat):
        '''
        Function which computes the gain of a 2-opt move.
        '''
        A, B, C, D = self.heuristic_path[i], self.heuristic_path[i+1],\
                     self.heuristic_path[j], self.heuristic_path[j+1]
        d1 = dist_mat[A,B] + dist_mat[C,D]
        d2 = dist_mat[A,C] + dist_mat[B,D]
        return d1 - d2

    def swap(self, i, j):
        '''
        Function which makes the 2-opt move.
        '''
        self.heuristic_path[i:j+1] = reversed(self.heuristic_path[i:j+1])



##### NEAREST NEIGHBOUR WITH 2-OPT SOLVER CLASS #####
class NN2Opt(TwoOpt):
    '''
    Nearest Neighbour with 2-opt class.
    '''

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 2-opt applied to NN.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        NN = NearestNeighbour()
        NN.solve(tsp)
        self.initial_path = NN.heuristic_path
        self.initial_cost = NN.heuristic_cost
        super().solve(tsp)



##### RANDOMIZED NEAREST NEIGHBOUR WITH 2-OPT SOLVER CLASS #####
class RNN2Opt(TwoOpt):
    '''
    Randomized Nearest Neighbour with 2-opt class.
    '''

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 2-opt applied to RNN.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        RNN = RandomizedNearestNeighbour()
        RNN.solve(tsp)
        self.initial_path = RNN.heuristic_path
        self.initial_cost = RNN.heuristic_cost
        super().solve(tsp)



##### 2-OPT + DLB SOLVER CLASS #####
class TwoOptDLB(TwoOpt):
    '''
    2-opt + Don't Look Bits class.
    '''

    def __init__(self, initial_path=None, initial_cost=None):
        super().__init__(initial_path=initial_path, initial_cost=initial_cost, dlb=True)

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 2-opt.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        super().solve(tsp)



##### NEAREST NEIGHBOUR WITH 2-OPT + DLB SOLVER CLASS #####
class NN2OptDLB(TwoOpt):
    '''
    Nearest Neighbour with 2-opt + Don't Look Bits class.
    '''

    def __init__(self):
        super().__init__(dlb=True)

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 2-opt applied to NN.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        NN = NearestNeighbour()
        NN.solve(tsp)
        self.initial_path = NN.heuristic_path
        self.initial_cost = NN.heuristic_cost
        super().solve(tsp)



##### RANDOMIZED NEAREST NEIGHBOUR WITH 2-OPT + DLB SOLVER CLASS #####
class RNN2OptDLB(TwoOpt):
    '''
    Randomized Nearest Neighbour with 2-opt + Don't Look Bits class.
    '''

    def __init__(self):
        super().__init__(dlb=True)

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 2-opt applied to RNN.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        RNN = RandomizedNearestNeighbour()
        RNN.solve(tsp)
        self.initial_path = RNN.heuristic_path
        self.initial_cost = RNN.heuristic_cost
        super().solve(tsp)



##### 3-OPT SOLVER CLASS #####
class ThreeOpt(Solver):
    '''
    Basic 3-opt algorithm.
    '''

    def __init__(self, initial_path=None, initial_cost=None):
        ''' 
        3-opt constructor.
        
        Parameters:
            - initial_node : int
                    The starting node for the solution.
            - initial_path : list
                    Initial path to which apply 3-opt algorithm
        '''
        super().__init__(initial_path=initial_path, initial_cost=initial_cost)

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 3-opt.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        self.heuristic_path = self.initial_path[:]
        self.heuristic_cost = self.initial_cost
        start_time = time.time()
        locally_optimal = False
        while not locally_optimal:
            locally_optimal = True
            for i in range(tsp.N - 4):
                if not locally_optimal: break
                for j in range(i + 2, tsp.N - 2):
                    if not locally_optimal: break
                    for k in range(j + 2, tsp.N - 1 if i == 0 else tsp.N):
                        case, gain = self.gain(i, j, k, tsp.dist_mat)
                        if gain > 0:
                            self.move(i, j, k, case)
                            self.heuristic_cost -= gain
                            locally_optimal = False
                            break
        end_time = time.time()
        self.heuristic_time = end_time - start_time

    def gain(self, i, j, k, dist_mat):
        '''
        Function which computes the gain of a 3-opt move.

        Parameters:
            - i, j, k: indexes of the nodes
            - dist_mat: matrix distance of the tsp instance.

        Returns the case and the gain of the opt move 
        which returns the minimum cost.
        '''
        A, B = self.heuristic_path[i], self.heuristic_path[i+1]
        C, D = self.heuristic_path[j], self.heuristic_path[j+1]
        E, F = self.heuristic_path[k], self.heuristic_path[k+1]
        
        d0 = dist_mat[A,B] + dist_mat[C,D] + dist_mat[E,F]
        d = [np.inf]*8 # distances vector
        d[1] = dist_mat[A,E] + dist_mat[C,D] + dist_mat[B,F]
        d[2] = dist_mat[A,B] + dist_mat[C,E] + dist_mat[D,F]
        d[3] = dist_mat[A,C] + dist_mat[B,D] + dist_mat[E,F]
        d[4] = dist_mat[A,C] + dist_mat[B,E] + dist_mat[D,F]
        d[5] = dist_mat[A,E] + dist_mat[B,D] + dist_mat[C,F]
        d[6] = dist_mat[A,D] + dist_mat[C,E] + dist_mat[B,F]
        d[7] = dist_mat[A,D] + dist_mat[B,E] + dist_mat[C,F]

        gain_vec = [d0 - i for i in d]
        case, gain = np.argmax(gain_vec), np.max(gain_vec)

        return case, gain
    
    def move(self, i, j, k, opt_case):
        '''
        Function which, given the case we are considering, 
        performs the 3-opt move.

        Parameters:
            - i, j, k: indexes of the 3-opt
            - opt_case: one of the 8 case which we are considering [0,1,...,6,7]
        '''
        # no change
        if opt_case == 0:
            pass
        # 2-opt moves
        elif opt_case == 1:
            self.swap(i+1,k)
        elif opt_case == 2:
            self.swap(j+1,k)
        elif opt_case == 3:
            self.swap(i+1,j)
        # 3-opt moves
        elif opt_case == 4:
            self.swap(i+1,j)
            self.swap(j+1,k)
        elif opt_case == 5:
            self.swap(i+1,j)
            self.swap(i+1,k)
        elif opt_case == 6:
            self.swap(j+1,k)
            self.swap(i+1,k)
        elif opt_case == 7:
            self.swap(i+1,j)
            self.swap(j+1,k)
            self.swap(i+1,k)

    def swap(self, i, j):
        '''
        Function which reverses the heuristic path from i-th component to j-th 
        component.

        swap([0,1,2,3,4,5,6], 2, 5) -> [0,1,5,4,3,2,6]
        '''
        self.heuristic_path[i:j+1] = reversed(self.heuristic_path[i:j+1])



##### NEAREST NEIGHBOUR WITH 3-OPT SOLVER CLASS #####
class NN3Opt(ThreeOpt):
    '''
    Nearest Neighbour with 3-opt class.
    '''

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 3-opt applied to NN.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        NN = NearestNeighbour()
        NN.solve(tsp)
        self.initial_path = NN.heuristic_path
        self.initial_cost = NN.heuristic_cost
        super().solve(tsp)



##### RNADOMIZED NEAREST NEIGHBOUR WITH 3-OPT SOLVER CLASS #####
class RNN3Opt(ThreeOpt):
    '''
    Randomized Nearest Neighbour with 3-opt class.
    '''

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 3-opt applied to RNN.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        RNN = RandomizedNearestNeighbour()
        RNN.solve(tsp)
        self.initial_path = RNN.heuristic_path
        self.initial_cost = RNN.heuristic_cost
        super().solve(tsp)