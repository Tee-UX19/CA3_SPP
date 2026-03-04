"""Simulated Annealing for SPP"""

import numpy as np

def run(problem,params,seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    #unpacking parameters
    T = params['T']
    alpha = params['alpha']
    max_iter = params['max_iter']
    lambda_penalty = params['lambda_penalty']
    
    n= problem.num_cols
    current= np.random.randint(2, size=n)  # Random initial solution
    current_score= problem.evaluate(current) + lambda_penalty * problem.penalty(current)
    
    best= current.copy()
    best_score= current_score
    best_feasible=None
    best_feasible_cost=np.inf
    
    history = []
    
    for i in range(max_iter):
        neighbor= current.copy()
        flip_index= np.random.randint(0,n)
        neighbor[flip_index] = 1 - neighbor[flip_index]  # Flip a random bit
        neighbor_score= problem.evaluate(neighbor) + lambda_penalty * problem.penalty(neighbor)
        
        delta = neighbor_score - current_score
        if delta < 0:
            current = neighbor
            current_score = neighbor_score
        else:
            prob = np.exp(-delta / T)
            if np.random.rand() < prob:
                current = neighbor
                current_score = neighbor_score
    
        if current_score < best_score:
            best = current.copy()
            best_score = current_score
            
        if problem.is_feasible(current):
            current_cost = problem.evaluate(current)
            if current_cost < best_feasible_cost:
                best_feasible = current.copy()
                best_feasible_cost = current_cost
        
        T= T * alpha
        history.append(best_feasible_cost if best_feasible is not None else np.inf)
    return {
        "best_feasible": best_feasible,
        "best_feasible_cost": best_feasible_cost,
        "feasible": best_feasible is not None,
        "history": history
    }
    
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.parser import parse_file
    from src.spp import SPP

    for dataset in ['sppnw41', 'sppnw42', 'sppnw43']:
        data = parse_file(f"data/{dataset}.txt")
        problem = SPP(**data)

        params = {
            'T': 10000.0,
            'alpha': 0.9999,
            'max_iter': 500000,
            'lambda_penalty': float(np.mean(problem.costs) * 1.5)
        }

        feasible_count = 0
        costs = []
        for i in range(10):
            r = run(problem, params, seed=i)
            if r['feasible']:
                feasible_count += 1
                costs.append(r['best_feasible_cost'])

        print(f"{dataset}: {feasible_count}/10 feasible, "
              f"mean cost: {np.mean(costs):.1f}" if costs else f"{dataset}: no feasible")