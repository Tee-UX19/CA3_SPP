"""Improved Binary Genetic Algorithm for SPP
Enhancements over Standard BGA (Chu & Beasley [1]):
  1. Pseudo-random initialisation  — Algorithm 2, p341
  2. Heuristic improvement operator — Algorithm 1, p331
  3. Stochastic ranking             — Runarsson & Yao [2], Pf=0.45
"""
import numpy as np


# ---------------------------------------------------------------------------
# Helper: separate fitness (cost) and unfitness (penalty) — kept distinct
# as required by C&B's ranking scheme
# ---------------------------------------------------------------------------

def _fitness(problem, sol):
    """Objective: total cost (lower = better)."""
    return float(np.dot(problem.costs, sol))

def _unfitness(problem, sol):
    """Constraint violation: sum of (coverage_count - 1)^2."""
    counts = problem.matrix @ sol
    return float(np.sum((counts - 1) ** 2))


# ---------------------------------------------------------------------------
# 1. Pseudo-random initialisation  (Algorithm 2, p341 in [1])
# ---------------------------------------------------------------------------

def _pseudo_random_init(problem, rng):
    """Build one individual greedily with randomness.

    Logic:
      - Start with empty solution; all rows uncovered.
      - Shuffle rows randomly (avoids bias to early rows).
      - For each uncovered row i: find the column j in alpha_i (columns
        covering row i) that covers ONLY uncovered rows (beta_j ⊆ U).
        Among those, pick the one minimising c_j / |beta_j ∩ U|.
        If no such 'safe' column exists, skip row i for now.
      - Result: a near-feasible starting solution.
    """
    n = problem.num_cols
    m = problem.num_rows
    sol = np.zeros(n, dtype=int)
    covered = np.zeros(m, dtype=int)   # coverage count per row

    # alpha[j] = set of rows covered by column j  (precomputed from matrix)
    # beta[i]  = set of columns that cover row i
    # Both derived from problem.matrix (shape: m x n, binary)

    row_order = rng.permutation(m)

    for i in row_order:
        if covered[i] >= 1:
            continue  # already covered

        # Columns that cover row i
        candidate_cols = np.where(problem.matrix[i, :] == 1)[0]

        best_j = -1
        best_score = np.inf

        for j in candidate_cols:
            if sol[j] == 1:
                continue  # already selected
            # Rows this column would cover
            rows_of_j = np.where(problem.matrix[:, j] == 1)[0]
            # 'Safe' means it doesn't over-cover anything already covered
            if np.any(covered[rows_of_j] >= 1):
                continue  # would cause over-coverage — skip
            # Score: cost per newly covered row (lower = better)
            new_coverage = int(np.sum(covered[rows_of_j] == 0))
            if new_coverage == 0:
                continue
            score = problem.costs[j] / new_coverage
            if score < best_score:
                best_score = score
                best_j = j

        if best_j != -1:
            sol[best_j] = 1
            covered += problem.matrix[:, best_j]

    return sol


# ---------------------------------------------------------------------------
# 2. Heuristic improvement operator  (Algorithm 1, p331 in [1])
# ---------------------------------------------------------------------------

def _heuristic_improve(problem, sol):
    """DROP redundant columns, then ADD columns to fix uncovered rows.

    DROP phase:
      Iterate over selected columns in random order. If every row covered
      by column j is already covered by at least one OTHER selected column
      (wi >= 2), remove j safely.

    ADD phase:
      For each uncovered row (in random order), find a column j that:
        - covers row i
        - only covers rows that are still uncovered (beta_j ⊆ U)
        - minimises c_j / |beta_j ∩ U|  (best cost-coverage ratio)
      If found, add it.
    """
    sol = sol.copy()
    n = problem.num_cols
    m = problem.num_rows

    # w[i] = number of selected columns covering row i
    w = (problem.matrix @ sol).astype(int)   # shape (m,)

    # --- DROP procedure ---
    selected = list(np.where(sol == 1)[0])
    np.random.shuffle(selected)   # random order to avoid bias

    for j in selected:
        rows_of_j = np.where(problem.matrix[:, j] == 1)[0]
        # Safe to drop if every row covered by j has w >= 2
        if np.all(w[rows_of_j] >= 2):
            sol[j] = 0
            w[rows_of_j] -= 1

    # --- ADD procedure ---
    uncovered = list(np.where(w == 0)[0])   # U
    np.random.shuffle(uncovered)            # random row order

    remaining = list(uncovered)  # V — rows still to be examined

    i_idx = 0
    while i_idx < len(remaining):
        i = remaining[i_idx]
        i_idx += 1

        if w[i] >= 1:
            # Already covered by a previous ADD step
            continue

        candidate_cols = np.where(problem.matrix[i, :] == 1)[0]

        best_j = -1
        best_score = np.inf

        for j in candidate_cols:
            if sol[j] == 1:
                continue
            rows_of_j = np.where(problem.matrix[:, j] == 1)[0]
            # Only add if column covers exclusively uncovered rows
            if np.any(w[rows_of_j] >= 1):
                continue  # would over-cover — skip
            new_cov = int(np.sum(w[rows_of_j] == 0))
            if new_cov == 0:
                continue
            score = problem.costs[j] / new_cov
            if score < best_score:
                best_score = score
                best_j = j

        if best_j != -1:
            sol[best_j] = 1
            newly_covered = np.where(problem.matrix[:, best_j] == 1)[0]
            w[newly_covered] += 1

    return sol


# ---------------------------------------------------------------------------
# 3. Stochastic ranking  (Runarsson & Yao [2], Section III)
# ---------------------------------------------------------------------------

def _stochastic_rank(fitnesses, unfitnesses, pf=0.45):
    """Return indices sorted by stochastic ranking (bubble sort variant).

    At each comparison of adjacent elements:
      - If both are feasible (unfitness == 0): compare by fitness only.
      - Else with prob Pf: compare by fitness (ignoring constraints).
      - Else: compare by unfitness.

    Lower fitness / unfitness = better.  Returns sorted index array
    (best first).
    """
    n = len(fitnesses)
    idx = np.arange(n)

    # Single pass of bubble sort — repeated n times (enough to stabilise)
    for _ in range(n):
        swapped = False
        for j in range(n - 1):
            a, b = idx[j], idx[j + 1]
            u_a, u_b = unfitnesses[a], unfitnesses[b]
            f_a, f_b = fitnesses[a], fitnesses[b]

            # Compare a vs b: should a come AFTER b? (i.e., b is better?)
            if u_a == 0 and u_b == 0:
                # Both feasible: rank purely by objective
                swap = f_a > f_b
            elif np.random.random() < pf:
                # Ignore constraints: rank by objective
                swap = f_a > f_b
            else:
                # Rank by unfitness (constraint violation)
                swap = u_a > u_b

            if swap:
                idx[j], idx[j + 1] = idx[j + 1], idx[j]
                swapped = True

        if not swapped:
            break

    return idx  # idx[0] = best individual


# ---------------------------------------------------------------------------
# Crossover and mutation (same as Standard BGA — uniform + static/adaptive)
# ---------------------------------------------------------------------------

def _crossover(p1, p2, rng):
    mask = rng.integers(0, 2, size=len(p1))
    return np.where(mask, p1, p2).copy()


def _mutate(child, n, rng, population=None):
    """Static mutation (rate 1/n) + adaptive mutation.

    Adaptive mutation triggers when:
      (a) child has no selected columns → set each bit with prob Ms/n
      (b) a column is selected in ALL individuals → flip it in the child
    """
    Ms = 3   # static mutation rate parameter (C&B default)

    # Static mutation: flip each bit with prob Ms/n
    mask = rng.random(size=n) < (Ms / n)
    child = np.where(mask, 1 - child, child)

    # Adaptive mutation
    if population is not None:
        # If child has zero columns selected
        if np.sum(child) == 0:
            child = (rng.random(size=n) < (Ms / n)).astype(int)

        # If any column is universally selected across population, flip it
        col_sums = population.sum(axis=0)   # shape (n,)
        pop_size = len(population)
        always_selected = np.where(col_sums == pop_size)[0]
        if len(always_selected) > 0:
            child[always_selected] = 1 - child[always_selected]

    return child


# ---------------------------------------------------------------------------
# Matching selection (C&B Section 4.4)
# ---------------------------------------------------------------------------

def _matching_select(population, fitnesses, unfitnesses, rng):
    """Select two parents using matching selection.

    P1: random individual.
    P2: if P1 is feasible, pick P2 by tournament on fitness.
         Otherwise, pick P2 to maximise compatibility:
         score = |rows(P1) ∪ rows(P2)| - |rows(P1) ∩ rows(P2)|
    """
    pop_size = len(population)
    idx1 = rng.integers(0, pop_size)
    p1 = population[idx1]

    if unfitnesses[idx1] == 0:
        # P1 feasible: select P2 by tournament (fitness-based)
        candidates = rng.choice(pop_size, size=min(5, pop_size), replace=False)
        best = candidates[np.argmin(fitnesses[candidates])]
        p2 = population[best]
    else:
        # Select P2 by compatibility score
        rows_p1 = set(np.where(p1 == 1)[0])  # columns selected in P1 — used as proxy
        # Actually: rows covered by P1
        covered_p1 = np.where(problem_ref.matrix @ p1 > 0)[0]

        best_score = -np.inf
        best_idx = rng.integers(0, pop_size)
        sample = rng.choice(pop_size, size=min(10, pop_size), replace=False)

        for idx2 in sample:
            if idx2 == idx1:
                continue
            p2_cand = population[idx2]
            covered_p2 = np.where(problem_ref.matrix @ p2_cand > 0)[0]
            union = len(np.union1d(covered_p1, covered_p2))
            inter = len(np.intersect1d(covered_p1, covered_p2))
            score = union - inter
            if score > best_score:
                best_score = score
                best_idx = idx2
        p2 = population[best_idx]

    return p1, p2


# ---------------------------------------------------------------------------
# Ranking replacement (4-subgroup scheme, C&B Section 4.5)
# ---------------------------------------------------------------------------

def _ranking_replace(population, fitnesses, unfitnesses, child, child_f, child_u):
    """Replace a population member using the G1→G2→G3→G4 scheme.

    Returns updated (population, fitnesses, unfitnesses) or unchanged if
    child is a duplicate.
    """
    pop_size = len(population)

    # Duplicate check
    for s in population:
        if np.array_equal(s, child):
            return population, fitnesses, unfitnesses, False  # discard

    f_c, u_c = child_f, child_u

    # Partition into subgroups
    G1 = [i for i in range(pop_size) if fitnesses[i] >= f_c and unfitnesses[i] >= u_c]
    G2 = [i for i in range(pop_size) if fitnesses[i] <  f_c and unfitnesses[i] >= u_c]
    G3 = [i for i in range(pop_size) if fitnesses[i] >= f_c and unfitnesses[i] <  u_c]
    G4 = [i for i in range(pop_size) if fitnesses[i] <  f_c and unfitnesses[i] <  u_c]

    for group in [G1, G2, G3, G4]:
        if group:
            # Replace member with worst unfitness (ties broken by worst fitness)
            victim = max(group, key=lambda i: (unfitnesses[i], fitnesses[i]))
            population[victim] = child.copy()
            fitnesses[victim]   = child_f
            unfitnesses[victim] = child_u
            return population, fitnesses, unfitnesses, True

    # No group found (child dominates all) — replace worst overall
    victim = int(np.argmax(unfitnesses))
    population[victim] = child.copy()
    fitnesses[victim]   = child_f
    unfitnesses[victim] = child_u
    return population, fitnesses, unfitnesses, True


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

# Module-level reference so _matching_select can access the problem
problem_ref = None


def run(problem, params, seed=None):
    global problem_ref
    problem_ref = problem

    rng = np.random.default_rng(seed)

    pop_size = params['pop_size']
    max_iter = params['max_iter']
    pf       = params.get('pf', 0.45)

    n = problem.num_cols

    # --- Initialisation: pseudo-random (Algorithm 2) ---
    population = np.array([
        _pseudo_random_init(problem, rng) for _ in range(pop_size)
    ])

    fitnesses   = np.array([_fitness(problem, s)   for s in population])
    unfitnesses = np.array([_unfitness(problem, s) for s in population])

    # Track best feasible solution
    best_feasible      = None
    best_feasible_cost = np.inf

    for i in range(pop_size):
        if unfitnesses[i] == 0 and fitnesses[i] < best_feasible_cost:
            best_feasible      = population[i].copy()
            best_feasible_cost = fitnesses[i]

    history = []

    for iteration in range(max_iter):

        # --- Stochastic ranking: sort population, pick parents from top half ---
        ranked_idx = _stochastic_rank(fitnesses, unfitnesses, pf)
        # Use ranked order for parent selection (sample from top half)
        top_half = ranked_idx[:max(2, pop_size // 2)]

        # --- Matching selection ---
        i1 = rng.choice(top_half)
        p1 = population[i1]
        u1 = unfitnesses[i1]

        if u1 == 0:
            # P1 feasible: tournament on fitness from top half
            candidates = rng.choice(top_half, size=min(5, len(top_half)), replace=False)
            i2 = candidates[np.argmin(fitnesses[candidates])]
        else:
            covered_p1 = np.where(problem.matrix @ p1 > 0)[0]
            best_score = -np.inf
            i2 = rng.choice(top_half)
            sample = rng.choice(top_half, size=min(10, len(top_half)), replace=False)
            for idx2 in sample:
                if idx2 == i1:
                    continue
                p2c = population[idx2]
                covered_p2 = np.where(problem.matrix @ p2c > 0)[0]
                score = len(np.union1d(covered_p1, covered_p2)) - \
                        len(np.intersect1d(covered_p1, covered_p2))
                if score > best_score:
                    best_score = score
                    i2 = idx2

        p2 = population[i2]

        # --- Crossover + mutation ---
        child = _crossover(p1, p2, rng)
        child = _mutate(child, n, rng, population=population)

        # --- Heuristic improvement operator (Algorithm 1) ---
        child = _heuristic_improve(problem, child)

        # --- Evaluate child ---
        child_f = _fitness(problem, child)
        child_u = _unfitness(problem, child)

        # --- Ranking replacement ---
        population, fitnesses, unfitnesses, inserted = _ranking_replace(
            population, fitnesses, unfitnesses, child, child_f, child_u
        )

        # Track best feasible
        if child_u == 0 and child_f < best_feasible_cost:
            best_feasible      = child.copy()
            best_feasible_cost = child_f

        history.append(best_feasible_cost if best_feasible is not None else np.inf)

    return {
        "best_feasible":      best_feasible,
        "best_feasible_cost": best_feasible_cost,
        "feasible":           best_feasible is not None,
        "history":            history
    }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.parser import parse_file
    from src.spp import SPP

    params = {
        'pop_size': 100,
        'max_iter': 50000,
        'pf':       0.45,
    }

    for dataset in ['sppnw41', 'sppnw42', 'sppnw43']:
        data    = parse_file(f"data/{dataset}.txt")
        problem = SPP(**data)

        feasible_count = 0
        costs = []
        for i in range(5):   # 5 runs for smoke test
            r = run(problem, params, seed=i)
            if r['feasible']:
                feasible_count += 1
                costs.append(r['best_feasible_cost'])

        mean_str = f"mean cost: {np.mean(costs):.1f}" if costs else "no feasible"
        print(f"{dataset}: {feasible_count}/5 feasible, {mean_str}")