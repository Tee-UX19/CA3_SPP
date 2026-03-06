"""Standard Binary Genetic Algorithm for SPP
"""
import numpy as np


def get_combined_fitness(problem, solution, lambda_penalty):
    return float(problem.evaluate(solution)) + \
           lambda_penalty * float(problem.penalty(solution))


def crossover(parent1, parent2):
    mask = np.random.randint(0, 2, size=len(parent1))
    return np.where(mask, parent1, parent2).copy()


def tournament_select(population, fitness):
    i, j = np.random.choice(len(population), size=2, replace=False)
    return population[i] if fitness[i] < fitness[j] else population[j]


def mutate(child, n):
    mask = np.random.random(size=n) < (1.0 / n)
    return np.where(mask, 1 - child, child)


def run(problem, params, seed=None):
    if seed is not None:
        np.random.seed(seed)

    pop_size       = params['pop_size']
    max_iter       = params['max_iter']
    lambda_penalty = params['lambda_penalty']

    n = problem.num_cols
    m = problem.num_rows

    
    population = (np.random.random(size=(pop_size, n)) < (1.0 / m)).astype(int)

    
    fitness = np.array([
        get_combined_fitness(problem, population[i], lambda_penalty)
        for i in range(pop_size)
    ])

    
    best_feasible      = None
    best_feasible_cost = np.inf

    for i in range(pop_size):
        if problem.is_feasible(population[i]):
            cost = problem.evaluate(population[i])
            if cost < best_feasible_cost:
                best_feasible      = population[i].copy()
                best_feasible_cost = cost

    history = []

    for iteration in range(max_iter):
    
        parent1 = tournament_select(population, fitness)
        parent2 = tournament_select(population, fitness)

    
        child = crossover(parent1, parent2)
        child = mutate(child, n)

        
        child_fitness = get_combined_fitness(problem, child, lambda_penalty)

        worst_idx = int(np.argmax(fitness))
        if child_fitness < fitness[worst_idx]:
            population[worst_idx] = child
            fitness[worst_idx]    = child_fitness

        if problem.is_feasible(child):
            cost = problem.evaluate(child)
            if cost < best_feasible_cost:
                best_feasible      = child.copy()
                best_feasible_cost = cost

        history.append(best_feasible_cost)

    return {
        "best_feasible":      best_feasible,
        "best_feasible_cost": best_feasible_cost,
        "feasible":           best_feasible is not None,
        "history":            history
    }


if __name__ == "__main__":
    import sys, os, time
    sys.path.append(".")
    from src.parser import parse_file
    from src.spp import SPP

    params = {
        'pop_size': 200,
        'max_iter': 100000,
        'lambda_penalty': None,
    }

    os.makedirs("results/raw", exist_ok=True)

    for dataset in ['sppnw41', 'sppnw42', 'sppnw43']:
        data    = parse_file(f"data/{dataset}.txt")
        problem = SPP(**data)
        params['lambda_penalty'] = float(np.mean(problem.costs) * 1.5)

        costs          = []
        feasible_count = 0
        out_file       = f"results/raw/standard_bga_{dataset}.txt"

        with open(out_file, "w") as f:
            f.write(f"algorithm=standard_bga, dataset={dataset}, "
                    f"pop_size={params['pop_size']}, "
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