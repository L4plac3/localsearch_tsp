from abc import ABC, abstractmethod
import time
import numpy as np
import random



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
class NN(Solver):
    ''' 
    Nearest Neighbour solver class, derived from solver. 
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



##### REPEATED NEAREST NEIGHBOUR SOLVER CLASS #####
class RepNN(NN):
    '''
    Repeated Nearest Neighbour solver class, derived from Nearest Neighbour.
    The Nearest Neighbour algorithm is applied to each node and then the best tour is picked.
    '''

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via Repeated Nearest Neighbour algorithm.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        start_time = time.time()
        for i in range(tsp.N):
            self.initial_node = i
            self.best_path = None
            self.best_cost = np.inf
            super().solve(tsp)
            if self.heuristic_cost < self.best_cost:
                self.best_path = self.heuristic_path
                self.best_cost = self.heuristic_cost
        self.heuristic_path = self.best_path
        self.heuristic_cost = self.best_cost
        end_time = time.time()
        self.heuristic_time = end_time - start_time



##### RANDOMIZED NEAREST NEIGHBOUR SOLVER CLASS #####
class RandNN(NN):
    ''' 
    Randomized Nearest Neighbour solver class, derived from Nearest Neighbour. 
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

    def __init__(self, initial_path=None, initial_cost=None, dlb=False, fixed_radius=False):
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
        self.fixed_radius = fixed_radius

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
                 
    def gain(self, i, j, dist_mat):
        '''
        Function which computes the gain of a 2-opt move.
        '''
        A, B, C, D = self.heuristic_path[i], self.heuristic_path[i+1],\
                     self.heuristic_path[j], self.heuristic_path[j+1]
        d1 = dist_mat[A,B] + dist_mat[C,D]
        d2 = dist_mat[A,C] + dist_mat[B,D]
        if self.fixed_radius:
            radius = dist_mat[A,B]
            if dist_mat[A,C] > radius:
                if dist_mat[B,D] > radius:
                    return -1
        return d1 - d2

    def swap(self, i, j):
        '''
        Function which makes the 2-opt move.
        '''
        self.heuristic_path[i:j+1] = reversed(self.heuristic_path[i:j+1])

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
        solver = NN()
        solver.solve(tsp)
        self.initial_path = solver.heuristic_path
        self.initial_cost = solver.heuristic_cost
        super().solve(tsp)



##### REPEATED NEAREST NEIGHBOUR WITH 2-OPT SOLVER CLASS #####
class RepNN2Opt(TwoOpt):
    '''
    Repeated Nearest Neighbour with 2-opt class.
    '''

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 2-opt applied to RepNN.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        solver = RepNN()
        solver.solve(tsp)
        self.initial_path = solver.heuristic_path
        self.initial_cost = solver.heuristic_cost
        super().solve(tsp)



##### RANDOMIZED NEAREST NEIGHBOUR WITH 2-OPT SOLVER CLASS #####
class RandNN2Opt(TwoOpt):
    '''
    Randomized Nearest Neighbour with 2-opt class.
    '''

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 2-opt applied to RandNN.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        solver = RandNN()
        solver.solve(tsp)
        self.initial_path = solver.heuristic_path
        self.initial_cost = solver.heuristic_cost
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
        solver = NN()
        solver.solve(tsp)
        self.initial_path = solver.heuristic_path
        self.initial_cost = solver.heuristic_cost
        super().solve(tsp)



##### REPEATED NEAREST NEIGHBOUR WITH 2-OPT + DLB SOLVER CLASS #####
class RepNN2OptDLB(TwoOpt):
    '''
    Repeated Nearest Neighbour with 2-opt + Don't Look Bits class.
    '''

    def __init__(self):
        super().__init__(dlb=True)

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 2-opt applied to RepNN.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        solver = RepNN()
        solver.solve(tsp)
        self.initial_path = solver.heuristic_path
        self.initial_cost = solver.heuristic_cost
        super().solve(tsp)



##### RANDOMIZED NEAREST NEIGHBOUR WITH 2-OPT + DLB SOLVER CLASS #####
class RandNN2OptDLB(TwoOpt):
    '''
    Randomized Nearest Neighbour with 2-opt + Don't Look Bits class.
    '''

    def __init__(self):
        super().__init__(dlb=True)

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 2-opt applied to RandNN.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        solver = RandNN()
        solver.solve(tsp)
        self.initial_path = solver.heuristic_path
        self.initial_cost = solver.heuristic_cost
        super().solve(tsp)



##### NEAREST NEIGHBOUR WITH 2-OPT + FIXED-RADIUS SOLVER CLASS #####
class NN2OptFR(TwoOpt):
    '''
    Nearest Neighbour with 2-opt + Fixed-Radius class.
    '''

    def __init__(self):
        super().__init__(fixed_radius=True)

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 2-opt applied to NN.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        solver = NN()
        solver.solve(tsp)
        self.initial_path = solver.heuristic_path
        self.initial_cost = solver.heuristic_cost
        super().solve(tsp)



##### REPEATED NEAREST NEIGHBOUR WITH 2-OPT + FIXED-RADIUS SOLVER CLASS #####
class RepNN2OptFR(TwoOpt):
    '''
    Repeated Nearest Neighbour with 2-opt + Fixed-Radius class.
    '''

    def __init__(self):
        super().__init__(fixed_radius=True)

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 2-opt applied to RepNN.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        solver = RepNN()
        solver.solve(tsp)
        self.initial_path = solver.heuristic_path
        self.initial_cost = solver.heuristic_cost
        super().solve(tsp)



##### RANDOMIZED NEAREST NEIGHBOUR WITH 2-OPT + FIXED-RADIUS SOLVER CLASS #####
class RandNN2OptFR(TwoOpt):
    '''
    Randomized Nearest Neighbour with 2-opt + Fixed-Radius class.
    '''

    def __init__(self):
        super().__init__(fixed_radius=True)

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 2-opt applied to RandNN.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        solver = RandNN()
        solver.solve(tsp)
        self.initial_path = solver.heuristic_path
        self.initial_cost = solver.heuristic_cost
        super().solve(tsp)



##### 3-OPT SOLVER CLASS #####
class ThreeOpt(Solver):
    '''
    Basic 3-opt algorithm.
    '''

    def __init__(self, initial_path=None, initial_cost=None, dlb=False):
        ''' 
        3-opt constructor.
        
        Parameters:
            - initial_node : int
                    The starting node for the solution.
            - initial_path : list
                    Initial path to which apply 3-opt algorithm
        '''
        super().__init__(initial_path=initial_path, initial_cost=initial_cost)
        self.dlb = dlb

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
        if self.dlb: self.setAllDLB(tsp.N+1, False)
        while not locally_optimal:
            locally_optimal = True
            for i in range(tsp.N - 4):
                if not locally_optimal: break
                if self.dlb:
                    if self.dlb_arr[i]: continue
                    node_improved = False
                for j in range(i + 2, tsp.N - 2):
                    if not locally_optimal: break
                    for k in range(j + 2, tsp.N - 1 if i == 0 else tsp.N):
                        case, gain = self.gain(i, j, k, tsp.dist_mat)
                        if gain > 0:
                            self.move(i, j, k, case)
                            self.heuristic_cost -= gain
                            locally_optimal = False
                            if self.dlb: 
                                self.setDLB([i,i+1,j,j+1,k,k+1],False)
                                node_improved = True
                            break
                    # if self.dlb and not node_improved: self.setDLB([j],True)
                if self.dlb and not node_improved: self.setDLB([i],True)
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
        solver = NN()
        solver.solve(tsp)
        self.initial_path = solver.heuristic_path
        self.initial_cost = solver.heuristic_cost
        super().solve(tsp)



##### REPEATED NEAREST NEIGHBOUR WITH 3-OPT SOLVER CLASS #####
class RepNN3Opt(ThreeOpt):
    '''
    Repeated Nearest Neighbour with 3-opt class.
    '''

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 3-opt applied to RepNN.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        solver = RepNN()
        solver.solve(tsp)
        self.initial_path = solver.heuristic_path
        self.initial_cost = solver.heuristic_cost
        super().solve(tsp)



##### RANDOMIZED NEAREST NEIGHBOUR WITH 3-OPT SOLVER CLASS #####
class RandNN3Opt(ThreeOpt):
    '''
    Randomized Nearest Neighbour with 3-opt class.
    '''

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 3-opt applied to RandNN.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        solver = RandNN()
        solver.solve(tsp)
        self.initial_path = solver.heuristic_path
        self.initial_cost = solver.heuristic_cost
        super().solve(tsp)



##### NEAREST NEIGHBOUR WITH 3-OPT + DLB SOLVER CLASS #####
class NN3OptDLB(ThreeOpt):
    '''
    Nearest Neighbour with 3-opt + Don't Look Bits class.
    '''

    def __init__(self):
        super().__init__(dlb=True)

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 3-opt applied to NN.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        solver = NN()
        solver.solve(tsp)
        self.initial_path = solver.heuristic_path
        self.initial_cost = solver.heuristic_cost
        super().solve(tsp)



##### REPEATED NEAREST NEIGHBOUR WITH 3-OPT + DLB SOLVER CLASS #####
class RepNN3OptDLB(ThreeOpt):
    '''
    Repeated Nearest Neighbour with 3-opt + Don't Look Bits class.
    '''

    def __init__(self):
        super().__init__(dlb=True)

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 3-opt applied to RepNN.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        solver = RepNN()
        solver.solve(tsp)
        self.initial_path = solver.heuristic_path
        self.initial_cost = solver.heuristic_cost
        super().solve(tsp)



##### RANDOMIZED NEAREST NEIGHBOUR WITH 3-OPT + DLB SOLVER CLASS #####
class RandNN3OptDLB(ThreeOpt):
    '''
    Randomized Nearest Neighbour with 3-opt + Don't Look Bits class.
    '''

    def __init__(self):
        super().__init__(dlb=True)

    def solve(self, tsp):
        ''' 
        Solve method for a given tsp instance via 2-opt applied to RandNN.
        
        Parameters:
            - tsp : tsp instance to solve
        '''
        solver = RandNN()
        solver.solve(tsp)
        self.initial_path = solver.heuristic_path
        self.initial_cost = solver.heuristic_cost
        super().solve(tsp)