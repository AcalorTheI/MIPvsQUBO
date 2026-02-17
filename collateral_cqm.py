"""
Collateral Optimisation using Constrained Quadratic Model (CQM)
================================================================

CQM reformulates the collateral allocation problem with *hard* constraints
instead of penalty terms.

Key advantages over QUBO:
  - Constraints are enforced natively, not as soft penalties
  - The solver filters infeasible solutions automatically
  - Much simpler formulation (no penalty weight tuning)

Two modes:
  "hybrid" — uses Real (continuous) variables with LeapHybridCQMSampler
             (D-Wave Leap cloud, native CQM support, best quality)
  "neal"   — uses Integer variables (lot-based discretisation) converted
             to BQM via dimod.cqm_to_bqm(), solved with Neal SA (local)

Formulation (hybrid / continuous):
  min  sum_ij( opportunity_cost[i] * x[i][j] )
  s.t. sum_i( (1-haircut[i]) * x[i][j] ) >= required[j]    for each obligation j
       sum_j( x[i][j] ) <= market_value[i]                  for each asset i
       x[i][j] >= 0,  x[i][j] = 0 for ineligible pairs

Formulation (neal / discretised):
  min  sum_ij( opportunity_cost[i] * lot_size * n[i][j] )
  s.t. sum_i( (1-haircut[i]) * lot_size * n[i][j] ) >= required[j]
       sum_j( n[i][j] ) <= max_lots[i]
       n[i][j] >= 0 integer,  n[i][j] = 0 for ineligible pairs

Backends:
  - "neal"   : CQM with Integer vars -> BQM -> Neal SA (local, no account)
  - "hybrid" : CQM with Real vars -> LeapHybridCQMSampler (Leap cloud)
"""

import warnings
import numpy as np
import dimod
import neal

from problem_data import ASSETS, OBLIGATIONS


# ---------------------------------------------------------------------------
# CQM CONSTRUCTION
# ---------------------------------------------------------------------------

def build_cqm(assets, obligations, mode="real", lot_size=1_000_000):
    """
    Build a dimod.ConstrainedQuadraticModel for the collateral problem.

    Parameters
    ----------
    assets       : list of asset dicts
    obligations  : list of obligation dicts
    mode         : "real" for continuous variables, "integer" for lot-based
    lot_size     : float — lot size for integer mode (ignored in real mode)

    Returns
    -------
    cqm      : dimod.ConstrainedQuadraticModel
    var_map  : dict mapping (i, j) -> variable label string
    scale    : float — multiply variable values by this to get dollar amounts
               (1.0 for real mode, lot_size for integer mode)
    """
    num_assets = len(assets)
    num_obligations = len(obligations)

    cqm = dimod.ConstrainedQuadraticModel()

    var_map = {}   # (i, j) -> label
    variables = {} # (i, j) -> dimod variable object

    if mode == "real":
        scale = 1.0
        for i in range(num_assets):
            for j, ob in enumerate(obligations):
                if i in ob["eligible_assets"]:
                    label = f"x_{i}_{j}"
                    var_map[(i, j)] = label
                    variables[(i, j)] = dimod.Real(
                        label,
                        lower_bound=0.0,
                        upper_bound=assets[i]["market_value"],
                    )

        # Objective: min sum( opp_cost[i] * x[i][j] )
        objective = 0.0
        for (i, j), var in variables.items():
            objective += assets[i]["opportunity_cost"] * var
        cqm.set_objective(objective)

        # Obligation requirements: sum_i( (1-h_i) * x[i][j] ) >= R_j
        for j, ob in enumerate(obligations):
            expr = 0.0
            for i in range(num_assets):
                if (i, j) in variables:
                    expr += (1.0 - assets[i]["haircut"]) * variables[(i, j)]
            cqm.add_constraint(expr >= ob["required_value"],
                               label=f"obligation_{j}")

        # Inventory limits: sum_j( x[i][j] ) <= MV_i
        for i in range(num_assets):
            expr = 0.0
            for j in range(num_obligations):
                if (i, j) in variables:
                    expr += variables[(i, j)]
            cqm.add_constraint(expr <= assets[i]["market_value"],
                               label=f"inventory_{i}")

    else:  # integer mode
        scale = lot_size
        max_lots = [int(a["market_value"] / lot_size) for a in assets]

        for i in range(num_assets):
            for j, ob in enumerate(obligations):
                if i in ob["eligible_assets"]:
                    label = f"n_{i}_{j}"
                    var_map[(i, j)] = label
                    variables[(i, j)] = dimod.Integer(
                        label,
                        lower_bound=0,
                        upper_bound=max_lots[i],
                    )

        # Objective: min sum( opp_cost[i] * lot_size * n[i][j] )
        objective = 0.0
        for (i, j), var in variables.items():
            objective += assets[i]["opportunity_cost"] * lot_size * var
        cqm.set_objective(objective)

        # Obligation requirements: sum_i( (1-h_i) * lot_size * n[i][j] ) >= R_j
        for j, ob in enumerate(obligations):
            expr = 0.0
            for i in range(num_assets):
                if (i, j) in variables:
                    expr += (1.0 - assets[i]["haircut"]) * lot_size * variables[(i, j)]
            cqm.add_constraint(expr >= ob["required_value"],
                               label=f"obligation_{j}")

        # Inventory limits: sum_j( n[i][j] ) <= max_lots[i]
        for i in range(num_assets):
            expr = 0.0
            for j in range(num_obligations):
                if (i, j) in variables:
                    expr += variables[(i, j)]
            cqm.add_constraint(expr <= max_lots[i],
                               label=f"inventory_{i}")

    return cqm, var_map, scale


# ---------------------------------------------------------------------------
# SAMPLER FACTORY
# ---------------------------------------------------------------------------

def _make_sampler(backend="neal"):
    """
    Create a sampler for the CQM.

    Returns
    -------
    sampler, is_native_cqm : (sampler, bool)
    """
    if backend == "neal":
        return neal.SimulatedAnnealingSampler(), False

    if backend == "hybrid":
        try:
            from dwave.system import LeapHybridCQMSampler
        except ImportError:
            raise ImportError(
                "CQM hybrid backend requires 'dwave-system'. "
                "Install with: pip install dwave-system\n"
                "Then configure your API token: dwave config create"
            )
        return LeapHybridCQMSampler(), True

    raise ValueError(f"Unknown backend '{backend}'. Use 'neal' or 'hybrid'.")


# ---------------------------------------------------------------------------
# DECODE SOLUTION
# ---------------------------------------------------------------------------

def decode_solution(sample, var_map, scale, assets, obligations):
    """Convert a CQM sample dict back into an allocation matrix."""
    num_assets = len(assets)
    num_obligations = len(obligations)
    allocation = np.zeros((num_assets, num_obligations))

    for (i, j), label in var_map.items():
        allocation[i, j] = max(0.0, float(sample.get(label, 0.0))) * scale

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

def solve_cqm(assets=None, obligations=None,
              backend="neal", lot_size=1_000_000, time_limit=None,
              num_reads=20, num_sweeps=5000, seed=42,
              lagrange_multiplier=None):
    """
    Solve collateral optimisation via CQM using a D-Wave Ocean SDK sampler.

    Parameters
    ----------
    assets       : list of asset dicts (default: problem_data.ASSETS)
    obligations  : list of obligation dicts (default: problem_data.OBLIGATIONS)
    backend      : str — "neal" (local SA via CQM->BQM) or "hybrid" (Leap cloud)
    lot_size     : float — lot size for integer discretisation (neal only)
    time_limit   : int or None — time limit in seconds (hybrid only, minimum 5)
    num_reads    : int — number of SA reads (neal only)
    num_sweeps   : int — number of sweeps per SA run (neal only)
    seed         : int — random seed (neal only)
    lagrange_multiplier : float or None — penalty weight for CQM->BQM conversion
                   (neal only). None lets dimod auto-calculate.

    Returns
    -------
    dict with keys: allocation, total_cost, success, assets, obligations,
                    num_vars, constraint_violations, backend, feasible_count,
                    total_count
    """
    if assets is None:
        assets = ASSETS
    if obligations is None:
        obligations = OBLIGATIONS

    sampler, is_native_cqm = _make_sampler(backend)

    if is_native_cqm:
        # Hybrid — use Real (continuous) variables, native CQM
        cqm, var_map, scale = build_cqm(assets, obligations, mode="real")
        sample_params = {}
        if time_limit is not None:
            sample_params["time_limit"] = time_limit
        sampleset = sampler.sample_cqm(cqm, **sample_params)
    else:
        # Neal — use Integer variables, convert CQM to BQM
        cqm, var_map, scale = build_cqm(assets, obligations,
                                         mode="integer", lot_size=lot_size)
        bqm_kwargs = {}
        if lagrange_multiplier is not None:
            bqm_kwargs["lagrange_multiplier"] = lagrange_multiplier
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Did not add constraint")
            warnings.filterwarnings("ignore", message="For constraints with fractional")
            bqm, inverter = dimod.cqm_to_bqm(cqm, **bqm_kwargs)

        bqm_sampleset = sampler.sample(
            bqm,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            seed=seed,
        )

        # Convert BQM samples back to CQM variable space
        cqm_samples = []
        for bqm_sample, energy in zip(bqm_sampleset.samples(), bqm_sampleset.record.energy):
            cqm_sample = inverter(dict(bqm_sample))
            cqm_samples.append(cqm_sample)

        # Find best feasible sample
        best_sample = None
        best_cost = float("inf")
        feasible_count = 0
        total_count = len(cqm_samples)

        for cqm_sample in cqm_samples:
            is_feasible = cqm.check_feasible(cqm_sample)
            if is_feasible:
                feasible_count += 1
            cost = sum(
                assets[i]["opportunity_cost"] * max(0.0, cqm_sample.get(label, 0.0)) * scale
                for (i, j), label in var_map.items()
            )
            if is_feasible and cost < best_cost:
                best_cost = cost
                best_sample = cqm_sample

        if best_sample is None:
            # No feasible sample — pick lowest cost anyway
            best_cost = float("inf")
            for cqm_sample in cqm_samples:
                cost = sum(
                    assets[i]["opportunity_cost"] * max(0.0, cqm_sample.get(label, 0.0)) * scale
                    for (i, j), label in var_map.items()
                )
                if cost < best_cost:
                    best_cost = cost
                    best_sample = cqm_sample

        sol = decode_solution(best_sample, var_map, scale, assets, obligations)
        sol["num_vars"] = len(var_map)
        sol["constraint_violations"] = []
        sol["backend"] = backend
        sol["feasible_count"] = feasible_count
        sol["total_count"] = total_count

        # Check constraint satisfaction
        for j, ob in enumerate(obligations):
            effective = sum(
                sol["allocation"][i, j] * (1 - assets[i]["haircut"])
                for i in range(len(assets))
            )
            shortfall = ob["required_value"] - effective
            if shortfall > 1.0:
                sol["constraint_violations"].append((ob["name"], shortfall))

        for i, asset in enumerate(assets):
            excess = sol["allocation"][i, :].sum() - asset["market_value"]
            if excess > 1.0:
                sol["constraint_violations"].append((f"Inventory {asset['name']}", excess))

        return sol

    # Filter feasible solutions (hybrid path)
    total_count = len(sampleset)
    feasible = sampleset.filter(lambda row: row.is_feasible)
    feasible_count = len(feasible)

    if feasible_count > 0:
        best_sample = feasible.first.sample
    else:
        best_sample = sampleset.first.sample

    sol = decode_solution(best_sample, var_map, scale, assets, obligations)

    # Check constraint satisfaction
    violations = []
    for j, ob in enumerate(obligations):
        effective = sum(
            sol["allocation"][i, j] * (1 - assets[i]["haircut"])
            for i in range(len(assets))
        )
        shortfall = ob["required_value"] - effective
        if shortfall > 1.0:
            violations.append((ob["name"], shortfall))

    for i, asset in enumerate(assets):
        excess = sol["allocation"][i, :].sum() - asset["market_value"]
        if excess > 1.0:
            violations.append((f"Inventory {asset['name']}", excess))

    sol["num_vars"] = len(var_map)
    sol["constraint_violations"] = violations
    sol["backend"] = backend
    sol["feasible_count"] = feasible_count
    sol["total_count"] = total_count
    return sol


def print_results(sol):
    """Pretty-print the CQM solution."""
    assets = sol["assets"]
    obligations = sol["obligations"]
    x = sol["allocation"]

    backend = sol.get("backend", "neal")
    backend_labels = {"neal": "CQM (Neal SA)", "hybrid": "CQM (Hybrid)"}
    label = backend_labels.get(backend, backend)

    print("=" * 80)
    print(f"{label} -- COLLATERAL OPTIMISATION RESULTS")
    print("=" * 80)
    print(f"\nTotal opportunity cost: ${sol['total_cost']:,.0f}")
    print(f"Variables: {sol['num_vars']}")
    print(f"Feasible samples: {sol['feasible_count']}/{sol['total_count']}")

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

    parser = argparse.ArgumentParser(description="Collateral Optimisation via CQM")
    parser.add_argument("--backend", choices=["neal", "hybrid"], default="neal",
                        help="Sampler backend (default: neal)")
    parser.add_argument("--lot-size", type=int, default=1_000_000,
                        help="Lot size for discretisation, neal only (default: 1000000)")
    parser.add_argument("--num-reads", type=int, default=20,
                        help="Number of SA reads, neal only (default: 20)")
    parser.add_argument("--num-sweeps", type=int, default=5000,
                        help="Sweeps per SA run, neal only (default: 5000)")
    parser.add_argument("--time-limit", type=int, default=None,
                        help="Time limit in seconds, hybrid only (default: auto)")
    parser.add_argument("--lagrange", type=float, default=None,
                        help="Lagrange multiplier for CQM->BQM, neal only (default: auto)")
    args = parser.parse_args()

    sol = solve_cqm(
        backend=args.backend,
        lot_size=args.lot_size,
        num_reads=args.num_reads,
        num_sweeps=args.num_sweeps,
        time_limit=args.time_limit,
        lagrange_multiplier=args.lagrange,
    )
    print_results(sol)
