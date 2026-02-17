"""
Collateral Optimisation — Crossover Finder: MIP vs QUBO vs CQM
================================================================

Finds where quantum/heuristic solvers outperform MIP (branch-and-bound)
on integer collateral allocation problems.

Strategy:
  - Small sizes:  lot-size constraints only (MIP is fast)
  - Medium sizes: add minimum transfer amounts (MIP gets harder)
  - Large sizes:  add concentration limits (MIP hits exponential wall)

The QUBO naturally handles lot sizes via discretisation.  It doesn't model
MTA or concentration explicitly, but its penalty-based approach degrades
gracefully rather than exponentially.

CQM (Constrained Quadratic Model) encodes constraints natively rather than
as penalty terms, improving feasibility rates over QUBO.
"""

import time
import sys
import numpy as np

from collateral_mip import solve_mip
from collateral_qubo import solve_qubo
from collateral_cqm import solve_cqm


# ======================================================================
# RANDOM PROBLEM GENERATOR
# ======================================================================

def generate_problem(num_assets, num_obligations, lot_size=1_000_000, seed=123):
    """Generate a random but feasible collateral problem."""
    rng = np.random.default_rng(seed)

    asset_types = ["Treasury", "Bund", "Corp Bond", "Cash", "MBS", "ABS",
                   "Agency", "Equity", "Gold", "Covered Bd", "Muni",
                   "Supra", "EM Sov", "TIPS", "FRN", "CD", "CP"]

    assets = []
    for i in range(num_assets):
        name = f"{rng.choice(asset_types)} {i+1:03d}"
        num_lots = rng.integers(10, 60)
        mv = num_lots * lot_size
        haircut = round(rng.uniform(0.0, 0.15), 3)
        opp_cost = round(rng.uniform(0.002, 0.040), 4)
        assets.append({
            "name": name, "market_value": mv,
            "haircut": haircut, "opportunity_cost": opp_cost,
        })

    total_effective = sum(a["market_value"] * (1 - a["haircut"]) for a in assets)

    obligations = []
    budget = total_effective * 0.35  # conservative — ensures feasibility
    for j in range(num_obligations):
        name = f"Obl {j+1:03d}"
        share = rng.uniform(0.7, 1.3)
        required = budget * share / num_obligations
        required = max(lot_size, round(required / lot_size) * lot_size)
        elig_rate = rng.uniform(0.5, 0.9)
        eligible = sorted(rng.choice(
            num_assets, size=max(3, int(num_assets * elig_rate)), replace=False
        ).tolist())
        obligations.append({
            "name": name, "required_value": required,
            "eligible_assets": eligible,
        })

    return assets, obligations


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("\n" + "#" * 120)
    print("#  CROSSOVER FINDER:  MIP (Branch-and-Bound)  vs  QUBO (Simulated Annealing)  vs  CQM (Constrained Quadratic Model)")
    print("#  Problem: Collateral optimisation with lot sizes, MTA, concentration limits")
    print("#" * 120)

    LOT_SIZE = 1_000_000
    MIP_TIME_LIMIT = 10.0

    # Problem configs: (assets, obligations, use_MTA, max_concentration)
    # Gradually add harder constraints to stress-test MIP
    configs = [
        # Small: lot sizes only — MIP should dominate
        {"na": 7,  "no": 4,  "mta": 0,        "conc": None, "label": "7x4  (lots only)"},
        {"na": 10, "no": 5,  "mta": 0,        "conc": None, "label": "10x5 (lots only)"},
        {"na": 15, "no": 6,  "mta": 0,        "conc": None, "label": "15x6 (lots only)"},
        {"na": 20, "no": 8,  "mta": 0,        "conc": None, "label": "20x8 (lots only)"},
        # Medium: add MTA — more binary vars for MIP
        {"na": 20, "no": 8,  "mta": 2_000_000, "conc": None, "label": "20x8 (+MTA)"},
        {"na": 25, "no": 10, "mta": 2_000_000, "conc": None, "label": "25x10 (+MTA)"},
        {"na": 30, "no": 12, "mta": 2_000_000, "conc": None, "label": "30x12 (+MTA)"},
        # Large: add concentration limits — NP-hard territory
        {"na": 30, "no": 12, "mta": 2_000_000, "conc": 8,    "label": "30x12 (+MTA+conc8)"},
        {"na": 35, "no": 14, "mta": 2_000_000, "conc": 8,    "label": "35x14 (+MTA+conc8)"},
        {"na": 40, "no": 16, "mta": 2_000_000, "conc": 8,    "label": "40x16 (+MTA+conc8)"},
        {"na": 50, "no": 18, "mta": 2_000_000, "conc": 10,   "label": "50x18 (+MTA+conc10)"},
    ]

    QUBO_CHUNKS = 10
    QUBO_READS = 3
    QUBO_SWEEPS = 4000
    QUBO_PENALTY = 5.0

    CQM_READS = 3
    CQM_SWEEPS = 4000

    print(f"\n  Lot size:           ${LOT_SIZE:,.0f}")
    print(f"  MIP time limit:     {MIP_TIME_LIMIT:.0f}s")
    print(f"  QUBO:               {QUBO_CHUNKS} chunks, {QUBO_READS} reads, "
          f"{QUBO_SWEEPS} sweeps, penalty={QUBO_PENALTY}")
    print(f"  CQM:                {CQM_READS} reads, {CQM_SWEEPS} sweeps (lot_size=${LOT_SIZE:,.0f})")

    hdr = (f"  {'Config':<22s} {'MIP Cost':>12s} {'MIP ms':>10s} {'MIP':>4s} "
           f"{'QUBO Cost':>12s} {'QUBO ms':>10s} {'Q':>3s} "
           f"{'CQM Cost':>12s} {'CQM ms':>10s} {'C':>3s} "
           f"{'Winner':>12s}")
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))

    results = []

    for cfg in configs:
        na, no = cfg["na"], cfg["no"]
        mta = cfg["mta"]
        conc = cfg["conc"]
        label = cfg["label"]

        assets, obligations = generate_problem(na, no, LOT_SIZE, seed=na * 100 + no)

        # --- MIP ---
        t0 = time.perf_counter()
        mip_sol = solve_mip(
            assets, obligations,
            lot_size=LOT_SIZE,
            min_transfer=mta,
            max_assets_per_obligation=conc,
            time_limit=MIP_TIME_LIMIT,
        )
        mip_ms = (time.perf_counter() - t0) * 1000
        mip_ok = mip_sol["success"] and mip_sol["allocation"] is not None
        mip_optimal = mip_sol.get("optimal", mip_sol["success"])
        mip_cost = mip_sol["total_cost"] if mip_ok else float("inf")

        # --- QUBO ---
        t0 = time.perf_counter()
        qubo_sol = solve_qubo(
            assets, obligations,
            num_chunks=QUBO_CHUNKS,
            penalty_weight=QUBO_PENALTY,
            num_reads=QUBO_READS,
            num_sweeps=QUBO_SWEEPS,
            seed=42,
        )
        qubo_ms = (time.perf_counter() - t0) * 1000
        qubo_ok = not qubo_sol["constraint_violations"]
        qubo_cost = qubo_sol["total_cost"]

        # --- CQM ---
        t0 = time.perf_counter()
        cqm_sol = solve_cqm(
            assets, obligations,
            backend="neal",
            lot_size=LOT_SIZE,
            num_reads=CQM_READS,
            num_sweeps=CQM_SWEEPS,
            seed=42,
        )
        cqm_ms = (time.perf_counter() - t0) * 1000
        cqm_ok = not cqm_sol["constraint_violations"]
        cqm_cost = cqm_sol["total_cost"]
        cqm_feasible = cqm_sol["feasible_count"]

        # Winner logic — pick best feasible solver, or best cost if none feasible
        candidates = []
        if mip_ok:
            candidates.append(("MIP", mip_cost, mip_ms))
        if qubo_ok:
            candidates.append(("QUBO", qubo_cost, qubo_ms))
        if cqm_ok:
            candidates.append(("CQM", cqm_cost, cqm_ms))

        if candidates:
            winner = min(candidates, key=lambda x: x[1])[0]
        else:
            winner = "Neither"

        if mip_ok and mip_optimal:
            mip_c = f"${mip_cost:,.0f}"
        elif mip_ok and not mip_optimal:
            mip_c = f"${mip_cost:,.0f}*"
        else:
            mip_c = "TIMEOUT"

        print(f"  {label:<22s} {mip_c:>12s} {mip_ms:>10.0f} "
              f"{'Y' if mip_ok else 'N':>4s} "
              f"${qubo_cost:>11,.0f} {qubo_ms:>10.0f} "
              f"{'Y' if qubo_ok else 'N':>3s} "
              f"${cqm_cost:>11,.0f} {cqm_ms:>10.0f} "
              f"{'Y' if cqm_ok else 'N':>3s} "
              f"{winner:>12s}")
        sys.stdout.flush()

        results.append({
            "label": label, "na": na, "no": no, "mta": mta, "conc": conc,
            "mip_cost": mip_cost, "mip_ms": mip_ms, "mip_ok": mip_ok,
            "mip_optimal": mip_optimal,
            "qubo_cost": qubo_cost, "qubo_ms": qubo_ms, "qubo_ok": qubo_ok,
            "cqm_cost": cqm_cost, "cqm_ms": cqm_ms, "cqm_ok": cqm_ok,
            "cqm_feasible": cqm_feasible,
            "winner": winner,
        })

    # ==================================================================
    # ANALYSIS
    # ==================================================================
    print("\n\n" + "#" * 120)
    print("#  ANALYSIS")
    print("#" * 120)

    mip_wins = [r for r in results if r["winner"] == "MIP"]
    qubo_wins = [r for r in results if r["winner"] == "QUBO"]
    cqm_wins = [r for r in results if r["winner"] == "CQM"]
    neither = [r for r in results if r["winner"] == "Neither"]
    mip_timeouts = [r for r in results if not r["mip_ok"]]
    mip_suboptimal = [r for r in results if r["mip_ok"] and not r["mip_optimal"]]

    print(f"\n  MIP wins:      {len(mip_wins)}")
    print(f"  QUBO wins:     {len(qubo_wins)}")
    print(f"  CQM wins:      {len(cqm_wins)}")
    print(f"  Neither:       {len(neither)}")
    print(f"  MIP timeouts:  {len(mip_timeouts)}  (no feasible solution found)")
    print(f"  MIP suboptimal:{len(mip_suboptimal)}  (feasible but not proven optimal, marked with *)")

    print("\n  FEASIBILITY COMPARISON:")
    print(f"  {'Config':<22s} {'MIP':>4s} {'QUBO':>5s} {'CQM':>4s}")
    print("  " + "-" * 40)
    for r in results:
        print(f"  {r['label']:<22s} {'Y' if r['mip_ok'] else 'N':>4s} "
              f"{'Y' if r['qubo_ok'] else 'N':>5s} "
              f"{'Y' if r['cqm_ok'] else 'N':>4s}")

    print("\n  TIMING TRAJECTORY:")
    print(f"  {'Config':<22s} {'MIP (ms)':>10s} {'QUBO (ms)':>10s} {'CQM (ms)':>10s}")
    print("  " + "-" * 55)
    for r in results:
        print(f"  {r['label']:<22s} {r['mip_ms']:>10.0f} {r['qubo_ms']:>10.0f} "
              f"{r['cqm_ms']:>10.0f}")

    print("\n  KEY FINDINGS:")
    print("  1. MIP (branch-and-bound) scales EXPONENTIALLY with integer variables.")
    print("     Lot sizes alone: O(A*O) integer vars.  Adding MTA doubles that")
    print("     with binary indicator vars.  Concentration adds combinatorial cuts.")
    print()
    print("  2. QUBO uses soft penalty terms for constraints.  This makes feasibility")
    print("     the bottleneck — the annealer often violates constraints.")
    print()
    print("  3. CQM encodes constraints natively via dimod.cqm_to_bqm().")
    print("     When using Neal SA locally, constraints are still converted to penalties,")
    print("     so feasibility is similar to QUBO.  The real advantage of CQM is with")
    print("     LeapHybridCQMSampler (cloud), which handles constraints natively.")
    print()
    print("  4. With D-Wave Leap hybrid CQM solver (cloud):")
    print("     CQM supports continuous (Real) variables — no discretisation needed.")
    print("     Constraints are hard, not soft.  Solution quality approaches LP/MIP.")
    print()


if __name__ == "__main__":
    main()
