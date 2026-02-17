"""
Collateral Optimisation using Mixed-Integer Programming (MIP)
==============================================================

Extends the LP formulation with realistic integer constraints that make
the problem NP-hard:

  1. LOT SIZES — each asset can only be transferred in discrete lot sizes
     (e.g. $1M per bond lot).  This turns continuous variables into integers.

  2. MINIMUM TRANSFER AMOUNT (MTA) — if any asset is allocated to an
     obligation, the amount must exceed a minimum threshold (or be zero).
     This is a big-M indicator constraint requiring binary variables.

  3. MAX CONCENTRATION — each obligation can receive collateral from at most
     K distinct assets (prevents operational complexity).

These constraints are common in real collateral management and transform
a polynomial-time LP into an NP-hard MIP.

We use scipy.optimize.milp (HiGHS branch-and-bound) to solve it.
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import eye as speye

from problem_data import ASSETS, OBLIGATIONS


def solve_mip(assets=None, obligations=None, lot_size=1_000_000,
              min_transfer=500_000, max_assets_per_obligation=None,
              time_limit=60.0):
    """
    Solve collateral optimisation as a MIP with lot-size and MTA constraints.

    Decision variables (all integer):
      n[i][j] = number of lots of asset i allocated to obligation j
                (so actual value = n[i][j] * lot_size)

    Binary indicator variables (for MTA):
      y[i][j] = 1 if asset i is used at all for obligation j, 0 otherwise

    Parameters
    ----------
    assets       : list of asset dicts
    obligations  : list of obligation dicts
    lot_size     : float — minimum transferable unit ($)
    min_transfer : float — if allocated, must be >= this amount
    max_assets_per_obligation : int or None — max distinct assets per obligation
    time_limit   : float — solver time limit in seconds

    Returns
    -------
    dict matching LP/QUBO solver output format
    """
    if assets is None:
        assets = ASSETS
    if obligations is None:
        obligations = OBLIGATIONS

    na = len(assets)
    no = len(obligations)

    # Max lots per asset
    max_lots = np.array([int(a["market_value"] / lot_size) for a in assets])
    min_lots_for_mta = max(1, int(np.ceil(min_transfer / lot_size)))

    # Variable layout:
    #   n[i][j] : integer lots  — indices [0, na*no)
    #   y[i][j] : binary indicator — indices [na*no, 2*na*no)
    num_n = na * no
    num_y = na * no
    num_vars = num_n + num_y

    def nidx(i, j):
        return i * no + j

    def yidx(i, j):
        return num_n + i * no + j

    # --- Objective: minimise opportunity cost ---
    # cost = sum( opp_cost[i] * lot_size * n[i][j] )
    # y variables have 0 cost
    c = np.zeros(num_vars)
    for i, asset in enumerate(assets):
        for j in range(no):
            c[nidx(i, j)] = asset["opportunity_cost"] * lot_size

    # --- Integrality: n = integer (1), y = binary (1 with bounds 0-1) ---
    integrality = np.ones(num_vars, dtype=int)  # all integer

    # --- Bounds ---
    lb = np.zeros(num_vars)
    ub = np.full(num_vars, np.inf)

    for i, asset in enumerate(assets):
        for j, ob in enumerate(obligations):
            if i in ob["eligible_assets"]:
                ub[nidx(i, j)] = max_lots[i]
            else:
                ub[nidx(i, j)] = 0  # ineligible
            # y is binary
            ub[yidx(i, j)] = 1 if i in ob["eligible_assets"] else 0

    # --- Constraints ---
    A_rows = []
    lb_cons = []
    ub_cons = []

    # (a) Obligation requirements:
    #     sum_i( (1-h_i) * lot_size * n[i][j] ) >= required[j]
    for j, ob in enumerate(obligations):
        row = np.zeros(num_vars)
        for i, asset in enumerate(assets):
            if i in ob["eligible_assets"]:
                row[nidx(i, j)] = (1.0 - asset["haircut"]) * lot_size
        A_rows.append(row)
        lb_cons.append(ob["required_value"])
        ub_cons.append(np.inf)

    # (b) Inventory limits:
    #     sum_j( n[i][j] ) <= max_lots[i]
    for i in range(na):
        row = np.zeros(num_vars)
        for j in range(no):
            row[nidx(i, j)] = 1.0
        A_rows.append(row)
        lb_cons.append(0)
        ub_cons.append(max_lots[i])

    # (c) Linking n and y (big-M):
    #     n[i][j] <= max_lots[i] * y[i][j]   (if y=0, n must be 0)
    #     n[i][j] >= min_lots_for_mta * y[i][j]  (if y=1, n >= min_lots)
    for i in range(na):
        for j, ob in enumerate(obligations):
            if i in ob["eligible_assets"]:
                # Upper link: n[i][j] - max_lots[i]*y[i][j] <= 0
                row_up = np.zeros(num_vars)
                row_up[nidx(i, j)] = 1.0
                row_up[yidx(i, j)] = -max_lots[i]
                A_rows.append(row_up)
                lb_cons.append(-np.inf)
                ub_cons.append(0)

                # Lower link: n[i][j] - min_lots*y[i][j] >= 0
                row_lo = np.zeros(num_vars)
                row_lo[nidx(i, j)] = 1.0
                row_lo[yidx(i, j)] = -min_lots_for_mta
                A_rows.append(row_lo)
                lb_cons.append(0)
                ub_cons.append(np.inf)

    # (d) Max concentration: sum_i y[i][j] <= K  for each obligation j
    if max_assets_per_obligation is not None:
        for j in range(no):
            row = np.zeros(num_vars)
            for i in range(na):
                row[yidx(i, j)] = 1.0
            A_rows.append(row)
            lb_cons.append(0)
            ub_cons.append(max_assets_per_obligation)

    A = np.array(A_rows)
    constraints = LinearConstraint(A, lb_cons, ub_cons)
    bounds = Bounds(lb, ub)

    # --- Solve ---
    options = {"time_limit": time_limit, "presolve": True}
    result = milp(c, integrality=integrality, bounds=bounds,
                  constraints=constraints, options=options)

    if not result.success:
        return {
            "allocation": None, "total_cost": None, "success": False,
            "assets": assets, "obligations": obligations,
            "message": result.message,
        }

    x = result.x
    allocation = np.zeros((na, no))
    for i in range(na):
        for j in range(no):
            allocation[i, j] = x[nidx(i, j)] * lot_size

    return {
        "allocation": allocation,
        "total_cost": result.fun,
        "success": True,
        "assets": assets,
        "obligations": obligations,
    }


def print_results(sol):
    """Pretty-print the MIP solution."""
    if not sol["success"]:
        print(f"MIP failed: {sol.get('message', 'unknown')}")
        return

    assets = sol["assets"]
    obligations = sol["obligations"]
    x = sol["allocation"]

    print("=" * 80)
    print("MIP (Branch-and-Bound) -- COLLATERAL OPTIMISATION RESULTS")
    print("=" * 80)
    print(f"\nMinimised total opportunity cost: ${sol['total_cost']:,.0f}\n")

    for j, ob in enumerate(obligations):
        print(f"\n--- {ob['name']} (required: ${ob['required_value']:,.0f}) ---")
        posted = 0.0
        for i, asset in enumerate(assets):
            alloc = x[i, j]
            if alloc > 0.5:
                h = asset["haircut"]
                eff = alloc * (1 - h)
                posted += eff
                print(f"  {asset['name']:20s}  allocated: ${alloc:>14,.0f}"
                      f"  (haircut {h:.0%})  effective: ${eff:>14,.0f}")
        print(f"  {'TOTAL effective':>20s}: ${posted:>14,.0f}")
    print()


if __name__ == "__main__":
    sol = solve_mip()
    print_results(sol)
