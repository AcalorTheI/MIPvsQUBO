"""
Collateral Optimisation using QUBO + D-Wave Ocean SDK
======================================================

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

Samplers:
  - "neal"    : D-Wave Neal simulated annealing (C++, local, no account needed)
  - "qpu"     : D-Wave Advantage quantum annealer (requires Leap account + API token)
  - "hybrid"  : D-Wave hybrid solver (classical-quantum, handles large problems)
"""

import numpy as np
import dimod
import neal

from problem_data import ASSETS, OBLIGATIONS


# ---------------------------------------------------------------------------
# SAMPLER FACTORY
# ---------------------------------------------------------------------------

def _make_sampler(backend="neal"):
    """
    Create a D-Wave sampler based on the requested backend.

    Parameters
    ----------
    backend : str
        "neal"   — local simulated annealing (dwave-neal, no cloud needed)
        "qpu"    — D-Wave Advantage quantum processing unit via Leap cloud
        "hybrid" — D-Wave hybrid classical-quantum solver (LeapHybridSampler)

    Returns
    -------
    sampler : dimod-compatible sampler
    """
    if backend == "neal":
        return neal.SimulatedAnnealingSampler()

    if backend == "qpu":
        try:
            from dwave.system import DWaveSampler, EmbeddingComposite
        except ImportError:
            raise ImportError(
                "D-Wave QPU backend requires 'dwave-system'. "
                "Install with: pip install dwave-system\n"
                "Then configure your API token: dwave config create"
            )
        return EmbeddingComposite(DWaveSampler())

    if backend == "hybrid":
        try:
            from dwave.system import LeapHybridSampler
        except ImportError:
            raise ImportError(
                "D-Wave hybrid backend requires 'dwave-system'. "
                "Install with: pip install dwave-system\n"
                "Then configure your API token: dwave config create"
            )
        return LeapHybridSampler()

    raise ValueError(f"Unknown backend '{backend}'. Use 'neal', 'qpu', or 'hybrid'.")


# ---------------------------------------------------------------------------
# QUBO CONSTRUCTION
# ---------------------------------------------------------------------------

def build_qubo(assets, obligations, num_chunks=10, penalty_weight=1.0):
    """
    Build the QUBO as a dimod.BinaryQuadraticModel for the discretised
    collateral problem.

    Parameters
    ----------
    assets       : list of asset dicts
    obligations  : list of obligation dicts
    num_chunks   : int — number of binary bits per (asset, obligation) pair
    penalty_weight : float — multiplier for constraint-violation penalties

    Returns
    -------
    bqm          : dimod.BinaryQuadraticModel — the QUBO model
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

    # We build Q as a dict-of-dicts for dimod
    linear = {}
    quadratic = {}

    def effective_chunk(i):
        return chunk_sizes[i] * (1.0 - assets[i]["haircut"])

    # (1) OBJECTIVE — opportunity cost (linear/diagonal)
    for v, (i, j, k) in enumerate(var_map):
        linear[v] = linear.get(v, 0.0) + assets[i]["opportunity_cost"] * chunk_sizes[i]

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

        idxs = [v for v, _ in vars_j]
        effs = [e for _, e in vars_j]

        # Diagonal: e_v^2 - 2*R_j*e_v
        for idx, eff in zip(idxs, effs):
            linear[idx] = linear.get(idx, 0.0) + base_penalty * (eff * eff - 2.0 * R_j * eff)

        # Off-diagonal (upper triangle): 2 * e_a * e_b
        for a_pos in range(len(idxs)):
            for b_pos in range(a_pos + 1, len(idxs)):
                key = (idxs[a_pos], idxs[b_pos])
                quadratic[key] = quadratic.get(key, 0.0) + base_penalty * 2.0 * effs[a_pos] * effs[b_pos]

    # (3) PENALTY — inventory limits
    #     P * (sum_bits - num_chunks)^2
    inv_penalty = base_penalty * 1.0

    for i in range(num_assets):
        vars_i = [v for v, (i2, j2, k2) in enumerate(var_map) if i2 == i]
        if len(vars_i) == 0:
            continue
        C_i = num_chunks
        for idx in vars_i:
            linear[idx] = linear.get(idx, 0.0) + inv_penalty * (1.0 - 2.0 * C_i)
        for a_pos in range(len(vars_i)):
            for b_pos in range(a_pos + 1, len(vars_i)):
                key = (vars_i[a_pos], vars_i[b_pos])
                quadratic[key] = quadratic.get(key, 0.0) + inv_penalty * 2.0

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

    return bqm, var_map, chunk_sizes


# ---------------------------------------------------------------------------
# DECODE SOLUTION
# ---------------------------------------------------------------------------

def decode_solution(sample, var_map, chunk_sizes, assets, obligations):
    """Convert a D-Wave sample dict back into an allocation matrix."""
    num_assets = len(assets)
    num_obligations = len(obligations)
    allocation = np.zeros((num_assets, num_obligations))

    for v, (i, j, k) in enumerate(var_map):
        if sample[v] == 1:
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
               penalty_weight=1.0, num_reads=20, num_sweeps=5000, seed=42,
               beta_range=None, beta_schedule_type="geometric",
               backend="neal", annealing_time=None, chain_strength=None,
               time_limit=None):
    """
    Solve collateral optimisation via QUBO using a D-Wave Ocean SDK sampler.

    Parameters
    ----------
    assets       : list of asset dicts (default: problem_data.ASSETS)
    obligations  : list of obligation dicts (default: problem_data.OBLIGATIONS)
    num_chunks   : int — binary bits per (asset, obligation) pair
    penalty_weight : float — multiplier for constraint-violation penalties
    num_reads    : int — number of independent SA/QPU runs (best returned)
    num_sweeps   : int — number of sweeps per SA run (neal only)
    seed         : int — random seed for reproducibility (neal only)
    beta_range   : tuple(float, float) or None — (beta_min, beta_max) inverse
                   temperature range. None lets Neal auto-calculate. (neal only)
    beta_schedule_type : str — "geometric" or "linear" temperature schedule
                   (neal only)
    backend      : str — sampler backend:
                   "neal"   — local simulated annealing (default, no cloud)
                   "qpu"    — D-Wave Advantage quantum annealer (Leap cloud)
                   "hybrid" — D-Wave hybrid classical-quantum solver (Leap cloud)
    annealing_time : int or None — QPU annealing time in microseconds (qpu only,
                   default 20us, max 2000us). Longer times can improve quality.
    chain_strength : float or None — coupling strength for embedding chains
                   (qpu only). None lets EmbeddingComposite auto-calculate.
    time_limit   : int or None — time limit in seconds (hybrid only, minimum 3).

    Returns
    -------
    dict with keys: allocation, total_cost, success, assets, obligations,
                    qubo_energy, num_vars, constraint_violations, sampleset,
                    backend
    """
    if assets is None:
        assets = ASSETS
    if obligations is None:
        obligations = OBLIGATIONS

    bqm, var_map, chunk_sizes = build_qubo(assets, obligations, num_chunks, penalty_weight)

    sampler = _make_sampler(backend)

    # Build sampler-specific parameters
    if backend == "neal":
        sample_params = {
            "num_reads": num_reads,
            "num_sweeps": num_sweeps,
            "seed": seed,
            "beta_schedule_type": beta_schedule_type,
        }
        if beta_range is not None:
            sample_params["beta_range"] = beta_range

    elif backend == "qpu":
        sample_params = {
            "num_reads": num_reads,
        }
        if annealing_time is not None:
            sample_params["annealing_time"] = annealing_time
        if chain_strength is not None:
            sample_params["chain_strength"] = chain_strength

    elif backend == "hybrid":
        sample_params = {}
        if time_limit is not None:
            sample_params["time_limit"] = time_limit

    sampleset = sampler.sample(bqm, **sample_params)

    # Extract best sample
    best_sample = sampleset.first.sample
    qubo_energy = sampleset.first.energy

    sol = decode_solution(best_sample, var_map, chunk_sizes, assets, obligations)

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
    sol["sampleset"] = sampleset
    sol["backend"] = backend
    return sol


def print_results(sol):
    """Pretty-print the QUBO solution."""
    assets = sol["assets"]
    obligations = sol["obligations"]
    x = sol["allocation"]

    backend = sol.get("backend", "neal")
    backend_labels = {"neal": "D-Wave Neal SA", "qpu": "D-Wave QPU", "hybrid": "D-Wave Hybrid"}
    label = backend_labels.get(backend, backend)

    print("=" * 80)
    print(f"QUBO ({label}) -- COLLATERAL OPTIMISATION RESULTS")
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
    import argparse

    parser = argparse.ArgumentParser(description="Collateral Optimisation via QUBO")
    parser.add_argument("--backend", choices=["neal", "qpu", "hybrid"], default="neal",
                        help="Sampler backend (default: neal)")
    parser.add_argument("--num-chunks", type=int, default=10,
                        help="Binary bits per (asset, obligation) pair (default: 10)")
    parser.add_argument("--num-reads", type=int, default=20,
                        help="Number of SA/QPU reads (default: 20)")
    parser.add_argument("--num-sweeps", type=int, default=5000,
                        help="Sweeps per SA run, neal only (default: 5000)")
    parser.add_argument("--penalty-weight", type=float, default=1.0,
                        help="Constraint penalty multiplier (default: 1.0)")
    parser.add_argument("--annealing-time", type=int, default=None,
                        help="QPU annealing time in microseconds, qpu only (default: 20)")
    parser.add_argument("--chain-strength", type=float, default=None,
                        help="Embedding chain strength, qpu only (default: auto)")
    parser.add_argument("--time-limit", type=int, default=None,
                        help="Time limit in seconds, hybrid only (default: auto)")
    args = parser.parse_args()

    sol = solve_qubo(
        num_chunks=args.num_chunks,
        num_reads=args.num_reads,
        num_sweeps=args.num_sweeps,
        penalty_weight=args.penalty_weight,
        backend=args.backend,
        annealing_time=args.annealing_time,
        chain_strength=args.chain_strength,
        time_limit=args.time_limit,
    )
    print_results(sol)
