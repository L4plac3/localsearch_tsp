from scipy.spatial.distance import pdist, squareform
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import random
import time





##### TSP-LOADER CLASS #####
class TSPloader():
    '''
    Class used to load a .tsp file into a numpy array containing 
    a numpy array (x,y), where x and y are coordinates of the nodes.
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
        self.path_len = len(nodes)


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
        for i in range(self.path_len):
            plt.plot(*self.nodes[[route[i], route[i+1]]].T, 'b')
        plt.show()


    def getResults(self):
        '''
        Get the costs and the elapsed times of all the routes computed via a solver.
        '''
        routes = {}
        for solver, route in self.routes.items():
            routes[solver] = {  'cost': route['cost'],\
                                'time': route['time']}
        return routes


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

    def __init__(self, initial_node=0, initial_path=None, initial_cost=None, iter_num=None):
        ''' 
        Base solver constructor.
        
        Parameters:
            - initial_node : int 
                    The starting node for the solution.
            - initial_path : permutation of the nodes
                    Initial path to which apply the local search algorithms
            - iter_num : int
                    Number of iterations for the local search algorithms
        '''
        self.initial_node = initial_node
        self.initial_path = initial_path
        self.initial_cost = initial_cost
        self.iter_num = iter_num

    @abstractmethod
    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        pass





##### NEAREST NEIGHBOUR SOLVER CLASS #####
class PointConnection(Solver):
    '''
    Solver which connects the nodes in the exact order in which they appear in the list.
    path = [0, 1, 2, ... , N-1, N, 0]
    '''

    def __init__(self, initial_node=0):
        ''' 
        PointConnection constructor.
        
        Parameters:
            - initial_node : int
                    The starting node for the solution.
        '''
        super().__init__(initial_node)
    
    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via PointConnectio.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        start_time = time.time()
        self.heuristic_path = tsp.path
        self.heuristic_cost = tsp.computeCost(tsp.path)
        end_time = time.time()
        self.heuristic_time = end_time - start_time






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
        while len(self.heuristic_path) < tsp.path_len:
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
        N = tsp.path_len
        self.initial_node = random.randint(0, N-1)
        super().solve(tsp)





##### 2-OPT SOLVER CLASS #####
class TwoOpt(Solver):
    '''
    Basic 2-opt algorithm.
    '''

    def __init__(self, initial_path=None, initial_cost=None, iter_num=500):
        ''' 
        2-opt constructor.
        
        Parameters:
            - initial_node : int
                    The starting node for the solution.
            - initial_path : list
                    Initial path to which apply 2-opt algorithm
            - iter_num : int
                    Number of iterations for the local 2-opt search
        '''
        super().__init__(initial_path=initial_path, initial_cost=initial_cost, iter_num=iter_num)

    
    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 2-opt.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        self.heuristic_path = self.initial_path[:]
        self.heuristic_cost = self.initial_cost
        start_time = time.time()
        for _ in range(self.iter_num):
            improved = False
            for i in range(1, tsp.path_len-3):
                for j in range(i + 2, tsp.path_len):
                    gain = self.gain(i, j, tsp.dist_mat)
                    if gain > 0:
                        improved = True
                        self.swap(i, j)
                        self.heuristic_cost -= gain
            if not improved: break
        end_time = time.time()
        self.heuristic_time = end_time - start_time
                    

    def gain(self, i, j, dist_mat):
        '''
        Function which computes the gain of a 2-opt move.
        '''
        A, B, C, D = self.heuristic_path[i], self.heuristic_path[i+1],\
                     self.heuristic_path[j], self.heuristic_path[j+1],
        d1 = dist_mat[A,B] + dist_mat[C,D]
        d2 = dist_mat[A,C] + dist_mat[B,D]
        return d1 - d2

    
    def swap(self, i, j):
        '''
        Function which makes the 2-opt move.
        '''
        self.heuristic_path[i+1:j+1] = reversed(self.heuristic_path[i+1:j+1])





##### NEAREST NEIGHBOUR WITH 2-OPT SOLVER CLASS #####
class TwoOptNN(TwoOpt):
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
class TwoOptRNN(TwoOpt):
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
    