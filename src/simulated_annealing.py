"""Simulated Annealing for SPP"""

import numpy as np

def run(problem,params,seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    
    T = params['T']
    alpha = params['alpha']
    max_iter = params['max_iter']
    lambda_penalty = params['lambda_penalty']
    
    n= problem.num_cols
    current= np.random.randint(2, size=n)  
    current_score= problem.evaluate(current) + lambda_penalty * problem.penalty(current)
    
    best= current.copy()
    best_score= current_score
    best_feasible=None
    best_feasible_cost=np.inf
    
    history = []
    
    for i in range(max_iter):
        neighbor= current.copy()
        flip_index= np.random.randint(0,n)
        neighbor[flip_index] = 1 - neighbor[flip_index] 
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
    import sys, os, time
    sys.path.append(".")
    from src.parser import parse_file
    from src.spp import SPP

    os.makedirs("results/raw", exist_ok=True)

    for dataset in ['sppnw41', 'sppnw42', 'sppnw43']:
        data    = parse_file(f"data/{dataset}.txt")
        problem = SPP(**data)

        params = {
            'T':              10000.0,
            'alpha':          0.9999,
            'max_iter':       300000,
            'lambda_penalty': float(np.mean(problem.costs) * 1.5)
        }

        costs          = []
        feasible_count = 0
        out_file       = f"results/raw/sa_{dataset}.txt"

        with open(out_file, "w") as f:
            f.write(f"algorithm=sa, dataset={dataset}, "
                    f"T={params['T']}, alpha={params['alpha']}, "
                    f"max_iter={params['max_iter']}, "
                    f"lambda={params['lambda_penalty']:.1f}\n\n")

            for seed in range(30):
                start = time.time()
                r     = run(problem, params, seed=seed)
                cost  = r['best_feasible_cost'] if r['feasible'] else float('inf')
                costs.append(cost)
                if r['feasible']:
                    feasible_count += 1

                line = (f"seed={seed:02d}, cost={cost:.1f}, "
                        f"feasible={r['feasible']}, time={time.time()-start:.1f}s")
                print(line, flush=True)
                f.write(line + "\n")
                f.flush()

            summary = (f"\nmean={np.mean(costs):.1f}, std={np.std(costs):.1f}, "
                       f"feasible={feasible_count}/30")
            print(summary)
            f.write(summary + "\n")

        print(f"Saved to {out_file}\n")