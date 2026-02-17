"""
Collateral Optimisation using QUBO + Simulated Annealing
=========================================================

QUBO (Quadratic Unconstrained Binary Optimisation) reformulates the collateral
allocation problem into a form solvable by quantum annealers (D-Wave), gate-
based quantum computers (via QAOA), or classical heuristics like simulated
annealing.

Key idea — discretisation:
  The LP has continuous decision variables (dollar amounts).  QUBO requires
  binary variables, so we discretise each allocation x[i][j] into K "chunks":

      x[i][j]  ~=  chunk_size * sum_{k=0}^{K-1} b[i][j][k]

  where b[i][j][k] in {0, 1}.  Each binary variable represents whether we
  allocate the k-th chunk of asset i to obligation j.

  This introduces a granularity trade-off: more bits = better approximation
  of the continuous optimum, but exponentially larger search space.

QUBO formulation:
  min  x^T Q x

  where x is the binary vector and Q encodes:
    1. Objective:  opportunity cost of each chunk (diagonal terms)
    2. Penalty for violating obligation requirements  (quadratic terms)
    3. Penalty for exceeding asset inventory          (quadratic terms)
    4. Eligibility is enforced structurally (ineligible variables excluded)

We solve the QUBO with simulated annealing (Neal-style), using a vectorised
numpy implementation for performance.
"""

import numpy as np

from problem_data import ASSETS, OBLIGATIONS


# ---------------------------------------------------------------------------
# QUBO CONSTRUCTION
# ---------------------------------------------------------------------------

def build_qubo(assets, obligations, num_chunks=10, penalty_weight=1.0):
    """
    Build the QUBO matrix for the discretised collateral problem.

    Parameters
    ----------
    assets       : list of asset dicts
    obligations  : list of obligation dicts
    num_chunks   : int — number of binary bits per (asset, obligation) pair
    penalty_weight : float — multiplier for constraint-violation penalties

    Returns
    -------
    Q            : np.ndarray — upper-triangular QUBO matrix
    var_map      : list of (asset_idx, obligation_idx, chunk_idx) per variable
    chunk_sizes  : np.ndarray — market-value per chunk for each asset
    """
    num_assets = len(assets)
    num_obligations = len(obligations)

    chunk_sizes = np.array([a["market_value"] / num_chunks for a in assets])

    # Build variable index map — only for eligible (asset, obligation) pairs
    var_map = []
    for i in range(num_assets):
        for j, ob in enumerate(obligations):
            if i in ob["eligible_assets"]:
                for k in range(num_chunks):
                    var_map.append((i, j, k))

    num_vars = len(var_map)
    Q = np.zeros((num_vars, num_vars))

    def effective_chunk(i):
        return chunk_sizes[i] * (1.0 - assets[i]["haircut"])

    # (1) OBJECTIVE — opportunity cost (diagonal)
    for v, (i, j, k) in enumerate(var_map):
        Q[v, v] += assets[i]["opportunity_cost"] * chunk_sizes[i]

    # Penalty scaling — must dominate the objective
    max_obj_per_chunk = max(
        a["opportunity_cost"] * chunk_sizes[ai]
        for ai, a in enumerate(assets)
    )
    avg_chunk_eff = np.mean([effective_chunk(i) for i in range(num_assets)])
    base_penalty = penalty_weight * max_obj_per_chunk / (avg_chunk_eff ** 2) * num_chunks * 10

    # (2) PENALTY — obligation requirements
    #     P * (sum(e_v * b_v) - R_j)^2
    for j, ob in enumerate(obligations):
        R_j = ob["required_value"]
        vars_j = [(v, effective_chunk(i2)) for v, (i2, j2, k2) in enumerate(var_map) if j2 == j]

        idxs = np.array([v for v, _ in vars_j])
        effs = np.array([e for _, e in vars_j])

        # Diagonal: e_v^2 - 2*R_j*e_v
        Q[idxs, idxs] += base_penalty * (effs * effs - 2.0 * R_j * effs)

        # Off-diagonal (upper triangle): 2 * e_a * e_b
        if len(idxs) > 1:
            outer = base_penalty * 2.0 * np.outer(effs, effs)
            for a_idx in range(len(idxs)):
                for b_idx in range(a_idx + 1, len(idxs)):
                    Q[idxs[a_idx], idxs[b_idx]] += outer[a_idx, b_idx]

    # (3) PENALTY — inventory limits
    #     P * (sum_bits - num_chunks)^2
    inv_penalty = base_penalty * 1.0

    for i in range(num_assets):
        vars_i = np.array([v for v, (i2, j2, k2) in enumerate(var_map) if i2 == i])
        if len(vars_i) == 0:
            continue
        C_i = num_chunks
        Q[vars_i, vars_i] += inv_penalty * (1.0 - 2.0 * C_i)
        for a_idx in range(len(vars_i)):
            for b_idx in range(a_idx + 1, len(vars_i)):
                Q[vars_i[a_idx], vars_i[b_idx]] += inv_penalty * 2.0

    return Q, var_map, chunk_sizes


# ---------------------------------------------------------------------------
# SIMULATED ANNEALING — VECTORISED
# ---------------------------------------------------------------------------

def simulated_annealing(Q, num_reads=20, num_sweeps=5000, seed=42):
    """
    Solve QUBO via simulated annealing with vectorised delta-energy
    computation.  Runs multiple SA chains in parallel via batch processing.

    Parameters
    ----------
    Q          : np.ndarray — QUBO matrix (upper triangular)
    num_reads  : int — number of independent SA runs (best returned)
    num_sweeps : int — number of temperature sweeps per run
    seed       : int — random seed

    Returns
    -------
    best_x     : np.ndarray of {0, 1}
    best_energy: float
    """
    rng = np.random.default_rng(seed)
    n = Q.shape[0]

    # Precompute full symmetric interaction matrix
    Q_full = Q + Q.T
    Q_diag = np.diag(Q).copy()

    # Temperature schedule (geometric)
    T_start = float(np.abs(Q).max()) * 2.0
    T_end = T_start * 1e-8
    temps = T_start * np.power(T_end / T_start, np.arange(num_sweeps) / max(num_sweeps - 1, 1))

    best_x = None
    best_energy = np.inf

    for _ in range(num_reads):
        x = rng.integers(0, 2, size=n).astype(np.float64)
        # Maintain h = Q_full @ x  for O(n) delta energy lookups
        h = Q_full @ x

        e = float(x @ Q @ x)

        for step in range(num_sweeps):
            T = temps[step]

            # Shuffle visit order
            order = rng.permutation(n)

            # Process in mini-batches to amortise Python overhead
            # Each batch: compute delta energies vectorised, accept/reject
            batch_size = min(n, 64)
            for start in range(0, n, batch_size):
                batch = order[start:start + batch_size]
                blen = len(batch)

                # Delta energy for flipping each bit in batch
                s = x[batch]
                # delta_if_0to1 = Q_diag[i] + h[i] - Q_full[i,i]*x[i]
                # = Q_diag[i] + (h[i] - Q_full[i,i]*s)
                delta_raw = Q_diag[batch] + h[batch] - Q_full[batch, batch] * s
                # If bit is currently 1, flip sign
                delta = np.where(s == 1, -delta_raw, delta_raw)

                # Metropolis acceptance
                accept = delta < 0
                if T > 1e-30:
                    rand_vals = rng.random(blen)
                    boltzmann = np.exp(np.clip(-delta / T, -500, 500))
                    accept = accept | (rand_vals < boltzmann)

                # Apply accepted flips one by one (must be sequential
                # to keep h consistent, but the accept decision is batched)
                for k in range(blen):
                    if accept[k]:
                        idx = batch[k]
                        old_val = x[idx]
                        new_val = 1.0 - old_val
                        x[idx] = new_val
                        # Update h incrementally: O(n)
                        diff = new_val - old_val
                        h += Q_full[:, idx] * diff
                        e += delta[k] if old_val == 0 else -delta_raw[k]

        if e < best_energy:
            best_energy = e
            best_x = x.copy()

    return best_x.astype(int), best_energy


# ---------------------------------------------------------------------------
# DECODE SOLUTION
# ---------------------------------------------------------------------------

def decode_solution(x_binary, var_map, chunk_sizes, assets, obligations):
    """Convert binary QUBO solution back into an allocation matrix."""
    num_assets = len(assets)
    num_obligations = len(obligations)
    allocation = np.zeros((num_assets, num_obligations))

    for v, (i, j, k) in enumerate(var_map):
        if x_binary[v] == 1:
            allocation[i, j] += chunk_sizes[i]

    total_cost = sum(
        assets[i]["opportunity_cost"] * allocation[i, j]
        for i in range(num_assets)
        for j in range(num_obligations)
    )

    return {
        "allocation": allocation,
        "total_cost": total_cost,
        "success": True,
        "assets": assets,
        "obligations": obligations,
    }


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def solve_qubo(assets=None, obligations=None, num_chunks=10,
               penalty_weight=1.0, num_reads=20, num_sweeps=5000, seed=42):
    """
    Solve collateral optimisation via QUBO + simulated annealing.

    Returns dict with keys: allocation, total_cost, success, assets,
    obligations, qubo_energy, num_vars, constraint_violations
    """
    if assets is None:
        assets = ASSETS
    if obligations is None:
        obligations = OBLIGATIONS

    Q, var_map, chunk_sizes = build_qubo(assets, obligations, num_chunks, penalty_weight)
    x_binary, qubo_energy = simulated_annealing(Q, num_reads, num_sweeps, seed)
    sol = decode_solution(x_binary, var_map, chunk_sizes, assets, obligations)

    # Check constraint satisfaction
    violations = []
    for j, ob in enumerate(obligations):
        effective = sum(
            sol["allocation"][i, j] * (1 - assets[i]["haircut"])
            for i, asset in enumerate(assets)
        )
        shortfall = ob["required_value"] - effective
        if shortfall > 1.0:
            violations.append((ob["name"], shortfall))

    for i, asset in enumerate(assets):
        excess = sol["allocation"][i, :].sum() - asset["market_value"]
        if excess > 1.0:
            violations.append((f"Inventory {asset['name']}", excess))

    sol["qubo_energy"] = qubo_energy
    sol["num_vars"] = len(var_map)
    sol["constraint_violations"] = violations
    return sol


def print_results(sol):
    """Pretty-print the QUBO solution."""
    assets = sol["assets"]
    obligations = sol["obligations"]
    x = sol["allocation"]

    print("=" * 80)
    print("QUBO (Simulated Annealing) -- COLLATERAL OPTIMISATION RESULTS")
    print("=" * 80)
    print(f"\nTotal opportunity cost: ${sol['total_cost']:,.0f}")
    print(f"QUBO energy: {sol['qubo_energy']:,.2f}")
    print(f"Binary variables: {sol['num_vars']}")

    if sol["constraint_violations"]:
        print(f"\n  ** CONSTRAINT VIOLATIONS ({len(sol['constraint_violations'])}) **")
        for name, amount in sol["constraint_violations"]:
            print(f"     {name}: shortfall/excess ${amount:,.0f}")
    else:
        print("\n  All constraints satisfied.")

    for j, ob in enumerate(obligations):
        print(f"\n--- {ob['name']} (required: ${ob['required_value']:,.0f}) ---")
        posted_value = 0.0
        for i, asset in enumerate(assets):
            alloc = x[i, j]
            if alloc > 1.0:
                haircut = asset["haircut"]
                effective = alloc * (1 - haircut)
                posted_value += effective
                print(f"  {asset['name']:20s}  MV allocated: ${alloc:>14,.0f}"
                      f"  (haircut {haircut:.0%})  effective: ${effective:>14,.0f}")
        print(f"  {'TOTAL effective':>20s}: ${posted_value:>14,.0f}")

    print("\n\n--- ASSET UTILISATION ---")
    for i, asset in enumerate(assets):
        total_used = x[i, :].sum()
        pct = total_used / asset["market_value"] * 100
        print(f"  {asset['name']:20s}  "
              f"used ${total_used:>14,.0f} / ${asset['market_value']:>14,.0f}  ({pct:.1f}%)")
    print()


if __name__ == "__main__":
    sol = solve_qubo(num_chunks=10, num_reads=20, num_sweeps=5000)
    print_results(sol)
