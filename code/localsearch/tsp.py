from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import numpy as np
import random



##### TSP-GENERATOR CLASS #####
class Generator():
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
class Loader():
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