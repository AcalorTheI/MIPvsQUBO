"""
Collateral Optimisation using Linear Programming
==================================================

Problem:
  A bank has a portfolio of assets (bonds, cash, equities) that it must post
  as collateral against multiple margin obligations (e.g. derivative trades
  cleared through CCPs, bilateral CSAs, repo agreements).

  Each obligation has:
    - A required collateral value (after haircuts)
    - A set of eligible asset types it will accept

  Each asset has:
    - A market value
    - A haircut (%) that reduces its effective value when pledged
    - An opportunity cost -- the "cheapest to deliver" idea: we prefer to post
      the asset that costs us the least to give up

  Goal:
    Minimise the total opportunity cost of the collateral allocation while
    satisfying every obligation and never exceeding available inventory.

This is a classic linear programming (LP) problem, widely used in real
collateral management systems (e.g. Calypso, Murex, CloudMargin).

We use scipy.optimize.linprog to solve it.
"""

import numpy as np
from scipy.optimize import linprog

from problem_data import ASSETS, OBLIGATIONS


def solve_lp(assets=None, obligations=None):
    """
    Solve the collateral optimisation problem using linear programming.

    Returns
    -------
    dict with keys:
        "allocation"      : np.ndarray of shape (num_assets, num_obligations)
                            market value allocated from asset i to obligation j
        "total_cost"      : float — minimised opportunity cost
        "success"         : bool
        "assets"          : list of asset dicts
        "obligations"     : list of obligation dicts
    """
    if assets is None:
        assets = ASSETS
    if obligations is None:
        obligations = OBLIGATIONS

    num_assets = len(assets)
    num_obligations = len(obligations)

    def idx(i, j):
        return i * num_obligations + j

    num_vars = num_assets * num_obligations

    # --- Objective: minimise opportunity cost ---
    c = np.zeros(num_vars)
    for i, asset in enumerate(assets):
        for j in range(num_obligations):
            c[idx(i, j)] = asset["opportunity_cost"]

    # --- Inequality constraints (Ax <= b) ---
    A_ub = []
    b_ub = []

    # Obligation requirements: -sum((1-h_i)*x_ij) <= -required_j
    for j, ob in enumerate(obligations):
        row = np.zeros(num_vars)
        for i, asset in enumerate(assets):
            if i in ob["eligible_assets"]:
                row[idx(i, j)] = -(1.0 - asset["haircut"])
        A_ub.append(row)
        b_ub.append(-ob["required_value"])

    # Inventory limits: sum_j(x_ij) <= market_value_i
    for i, asset in enumerate(assets):
        row = np.zeros(num_vars)
        for j in range(num_obligations):
            row[idx(i, j)] = 1.0
        A_ub.append(row)
        b_ub.append(asset["market_value"])

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # --- Bounds ---
    bounds = []
    for i, asset in enumerate(assets):
        for j, ob in enumerate(obligations):
            if i in ob["eligible_assets"]:
                bounds.append((0, asset["market_value"]))
            else:
                bounds.append((0, 0))

    # --- Solve ---
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    allocation = result.x.reshape(num_assets, num_obligations) if result.success else None

    return {
        "allocation": allocation,
        "total_cost": result.fun if result.success else None,
        "success": result.success,
        "assets": assets,
        "obligations": obligations,
    }


def print_results(sol):
    """Pretty-print the LP solution."""
    assets = sol["assets"]
    obligations = sol["obligations"]
    x = sol["allocation"]

    print("=" * 80)
    print("LP — COLLATERAL OPTIMISATION RESULTS")
    print("=" * 80)
    print(f"\nMinimised total opportunity cost: ${sol['total_cost']:,.0f}\n")

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

    print("\n--- UNENCUMBERED (FREE) ASSETS ---")
    for i, asset in enumerate(assets):
        free = asset["market_value"] - x[i, :].sum()
        if free > 1.0:
            print(f"  {asset['name']:20s}  ${free:>14,.0f}")
    print()


# Allow standalone execution
if __name__ == "__main__":
    sol = solve_lp()
    if sol["success"]:
        print_results(sol)
    else:
        print("LP optimisation failed.")
