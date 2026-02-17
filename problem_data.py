"""
Shared problem data for collateral optimisation benchmarks.

Both the LP and QUBO solvers import from here so they solve the exact same
problem instance.
"""

# ---------------------------------------------------------------------------
# ASSET INVENTORY
# ---------------------------------------------------------------------------
ASSETS = [
    {"name": "US Treasury 2Y",   "market_value": 50_000_000, "haircut": 0.02, "opportunity_cost": 0.005},
    {"name": "US Treasury 10Y",  "market_value": 30_000_000, "haircut": 0.05, "opportunity_cost": 0.010},
    {"name": "German Bund 5Y",   "market_value": 20_000_000, "haircut": 0.03, "opportunity_cost": 0.008},
    {"name": "Corporate Bond A", "market_value": 25_000_000, "haircut": 0.10, "opportunity_cost": 0.025},
    {"name": "Cash USD",         "market_value": 40_000_000, "haircut": 0.00, "opportunity_cost": 0.035},
    {"name": "Cash EUR",         "market_value": 15_000_000, "haircut": 0.00, "opportunity_cost": 0.030},
    {"name": "Equity ETF",       "market_value": 10_000_000, "haircut": 0.15, "opportunity_cost": 0.040},
]

# ---------------------------------------------------------------------------
# OBLIGATIONS (margin calls / collateral requirements)
# ---------------------------------------------------------------------------
OBLIGATIONS = [
    {
        "name": "CCP LCH - IRS Portfolio",
        "required_value": 35_000_000,
        "eligible_assets": [0, 1, 2, 4, 5],  # govt bonds + cash only
    },
    {
        "name": "CCP CME - Futures",
        "required_value": 20_000_000,
        "eligible_assets": [0, 1, 4],  # US treasuries + USD cash
    },
    {
        "name": "Bilateral CSA - Counterparty X",
        "required_value": 15_000_000,
        "eligible_assets": [0, 1, 2, 3, 4, 5, 6],  # accepts everything
    },
    {
        "name": "Repo Agreement - Bank Y",
        "required_value": 10_000_000,
        "eligible_assets": [0, 1, 2, 3],  # bonds only
    },
]
