"""Improved BGA with pseudo-random init, stochastic ranking, heuristic improvement"""
import numpy as np

def pseudo_random_init(problem, pop_size):
    n          = problem.num_cols
    m          = problem.num_rows
    population = np.zeros((pop_size, n), dtype=int)

    for p in range(pop_size):
        solution  = np.zeros(n, dtype=int)
        U         = set(range(m))         

        while U:
            i = np.random.choice(list(U))

            cols_covering_i = np.where(problem.matrix[i, :] == 1)[0]
            candidates = []
            for j in cols_covering_i:
                if solution[j] == 0:
                    rows_of_j = set(np.where(problem.matrix[:, j] == 1)[0])
                    if rows_of_j.issubset(U):  
                        candidates.append(j)

            if candidates:
                j = np.random.choice(candidates)
                solution[j] = 1
                covered = set(np.where(problem.matrix[:, j] == 1)[0])
                U -= covered
            else:
                U.remove(i)

        population[p] = solution
    return population

def heuristic_improvement(problem, solution):
    sol      = solution.copy()
    coverage = problem.matrix @ sol

    selected = list(np.where(sol == 1)[0])
    np.random.shuffle(selected)
    for j in selected:
        rows_j = np.where(problem.matrix[:, j] == 1)[0]
        if np.all(coverage[rows_j] >= 2):
            sol[j]    = 0
            coverage -= problem.matrix[:, j]

    U         = set(np.where(coverage == 0)[0])
    V         = list(U)
    np.random.shuffle(V)

    for i in V:
        if i not in U:
            continue 

        cols_i     = np.where(problem.matrix[i, :] == 1)[0]
        candidates = []
        for j in cols_i:
            if sol[j] == 0:
                rows_j = set(np.where(problem.matrix[:, j] == 1)[0])
                if rows_j.issubset(U):   
                    candidates.append(j)

        if not candidates:
            continue

        best_j    = None
        best_ratio = np.inf
        for j in candidates:
            rows_j      = np.where(problem.matrix[:, j] == 1)[0]
            covered_in_U = sum(1 for r in rows_j if r in U)
            ratio        = problem.costs[j] / covered_in_U
            if ratio < best_ratio:
                best_ratio = ratio
                best_j     = j

        sol[best_j]  = 1
        covered       = set(np.where(problem.matrix[:, best_j] == 1)[0])
        coverage     += problem.matrix[:, best_j]
        U            -= covered

    return sol


def stochastic_ranking(fitness, unfitness, Pf=0.45):
    n       = len(fitness)
    indices = np.arange(n)

    for _ in range(n):
        swapped = False
        for i in range(n - 1):
            a, b = indices[i], indices[i + 1]
            u    = np.random.random()

            if (unfitness[a] == 0 and unfitness[b] == 0) or u < Pf:
                if fitness[a] > fitness[b]:
                    indices[i], indices[i + 1] = b, a
                    swapped = True
            else:
                if unfitness[a] > unfitness[b]:
                    indices[i], indices[i + 1] = b, a
                    swapped = True

        if not swapped:
            break

    return indices


def run(problem, params, seed=None):
    if seed is not None:
        np.random.seed(seed)

    pop_size = params['pop_size']
    max_iter = params['max_iter']
    Pf       = params.get('Pf', 0.45)
    n        = problem.num_cols
    m        = problem.num_rows

    population = pseudo_random_init(problem, pop_size)

    fitness   = np.array([float(problem.evaluate(population[i]))
                           for i in range(pop_size)])
    unfitness = np.array([float(problem.penalty(population[i]))
                           for i in range(pop_size)])

    best_feasible      = None
    best_feasible_cost = np.inf

    for i in range(pop_size):
        if unfitness[i] == 0 and fitness[i] < best_feasible_cost:
            best_feasible      = population[i].copy()
            best_feasible_cost = fitness[i]

    history = []

    for iteration in range(max_iter):
        ranked = stochastic_ranking(fitness, unfitness, Pf)

        top_half = ranked[:pop_size // 2]
        p1_idx   = top_half[np.random.randint(len(top_half))]
        p2_idx   = top_half[np.random.randint(len(top_half))]

        parent1 = population[p1_idx]
        parent2 = population[p2_idx]

        mask  = np.random.randint(0, 2, size=n)
        child = np.where(mask, parent1, parent2).copy()

        mut_mask = np.random.random(n) < (1.0 / n)
        child    = np.where(mut_mask, 1 - child, child)

        child_unfit = float(problem.penalty(child))
        if child_unfit > 0:
            adapt_rate = child_unfit / (n * m)
            adapt_mask = np.random.random(n) < adapt_rate
            child      = np.where(adapt_mask, 1 - child, child)

        child = heuristic_improvement(problem, child)

        if any(np.array_equal(child, population[i]) for i in range(pop_size)):
            history.append(best_feasible_cost)
            continue

        
        child_fitness   = float(problem.evaluate(child))
        child_unfitness = float(problem.penalty(child))

        worst = int(np.argmax(unfitness * 1e9 + fitness))
        population[worst] = child
        fitness[worst]    = child_fitness
        unfitness[worst]  = child_unfitness

        if child_unfitness == 0 and child_fitness < best_feasible_cost:
            best_feasible      = child.copy()
            best_feasible_cost = child_fitness

        history.append(best_feasible_cost)

    return {
        "best_feasible":      best_feasible,
        "best_feasible_cost": best_feasible_cost,
        "feasible":           best_feasible is not None,
        "history":            history
    }

#note that unlike the other files, we run this on each dataset separately. so python src/improved_bga.py sppnw41, then python src/improved_bga.py sppnw42, etc.

if __name__ == "__main__":
    import sys, os, time
    sys.path.append(".")
    from src.parser import parse_file
    from src.spp import SPP

    dataset = sys.argv[1] if len(sys.argv) > 1 else 'sppnw41'
    params  = {'pop_size': 100, 'max_iter': 20000, 'Pf': 0.45}

    data    = parse_file(f"data/{dataset}.txt")
    problem = SPP(**data)

    costs          = []
    feasible_count = 0

    os.makedirs("results/raw", exist_ok=True)
    out_file = f"results/raw/improved_bga_{dataset}.txt"

    with open(out_file, "w") as f:
        f.write(f"algorithm=improved_bga, dataset={dataset}, "
                f"pop_size={params['pop_size']}, max_iter={params['max_iter']}\n\n")

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

    print(f"Saved to {out_file}")
