"""SPP problem class - cost evaluation, feasibility checking, penalty calculation"""
import numpy as np

class SPP:
    def __init__(self, num_rows,num_cols,costs, matrix):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.costs = costs
        self.matrix = matrix
       
    # Evaluate the cost of a solution
    def evaluate(self, solution):
        return float(np.dot(self.costs,solution))
    
    # Calculate the coverage count for each row given a solution
    def coverage_count(self, solution):
        return self.matrix @ solution
    
    
    def is_feasible(self, solution):
        counts= self.coverage_count(solution)
        return bool(np.all(counts == 1))
    
    def penalty(self, solution):
        counts= self.coverage_count(solution)
        return float (np.sum((counts-1)**2))
    
