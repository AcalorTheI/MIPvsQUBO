# Collateral Optimisation

Solves the bank collateral allocation problem using three approaches: Linear Programming (LP), Mixed-Integer Programming (MIP), and Quadratic Unconstrained Binary Optimisation (QUBO) with simulated annealing. Includes a crossover benchmark comparing MIP vs QUBO at scale.

## Problem

A bank holds a portfolio of assets (bonds, cash, equities) that must be posted as collateral against multiple margin obligations (CCP trades, bilateral CSAs, repos). Each obligation has a required value and a set of eligible asset types. Each asset has a market value, a regulatory haircut, and an opportunity cost. The goal is to minimise total opportunity cost while satisfying all obligations and never exceeding available inventory.

## Requirements

- Python 3.8+
- NumPy
- SciPy

```
pip install numpy scipy
```

No other dependencies are required. The QUBO solver uses a pure-Python simulated annealing implementation.

## Project Structure

| File | Description |
|---|---|
| `problem_data.py` | Shared asset inventory and obligation definitions |
| `collateral_optimisation.py` | LP solver using `scipy.optimize.linprog` (HiGHS) |
| `collateral_mip.py` | MIP solver using `scipy.optimize.milp` (HiGHS branch-and-bound) |
| `collateral_qubo.py` | QUBO solver with simulated annealing |
| `main.py` | Crossover benchmark: MIP vs QUBO at increasing problem sizes |
| `bcbs189.pdf` | Basel III regulatory framework (BCBS 189) reference document |

## How to Run

### LP Solver (continuous relaxation)

```
python collateral_optimisation.py
```

Solves the problem with continuous decision variables. Fastest and produces the optimal lower bound. No configurable parameters at the command line; edit `problem_data.py` to change the problem instance.

### MIP Solver (integer lots + MTA + concentration)

```
python collateral_mip.py
```

Adds realistic integer constraints on top of the LP. Parameters are configured in `solve_mip()`:

| Parameter | Default | Description |
|---|---|---|
| `lot_size` | `1_000_000` | Minimum transferable unit in dollars. Assets can only be allocated in multiples of this amount. |
| `min_transfer` | `500_000` | Minimum transfer amount. If any asset is allocated to an obligation, the amount must be at least this value (or zero). Requires binary indicator variables. |
| `max_assets_per_obligation` | `None` | Maximum number of distinct assets that can be posted to a single obligation. `None` disables this constraint. |
| `time_limit` | `60.0` | Solver time limit in seconds. HiGHS branch-and-bound will return the best solution found within this budget. |

### QUBO Solver (simulated annealing)

```
python collateral_qubo.py
```

Reformulates the problem into a QUBO matrix and solves via simulated annealing. Parameters are configured in `solve_qubo()`:

| Parameter | Default | Description |
|---|---|---|
| `num_chunks` | `10` | Number of binary bits per (asset, obligation) pair. Each chunk represents `market_value / num_chunks` dollars. Higher values give better precision but exponentially larger search space. |
| `penalty_weight` | `1.0` | Multiplier for constraint-violation penalties. Higher values force feasibility at the expense of objective quality. |
| `num_reads` | `20` | Number of independent SA runs. The best solution across all runs is returned. |
| `num_sweeps` | `5000` | Number of temperature sweeps per SA run. Each sweep visits every variable once. More sweeps allow better convergence. |
| `seed` | `42` | Random seed for reproducibility. |

**QUBO variable count**: For `A` assets, `O` obligations, and `K` chunks, the number of binary variables is up to `A * O * K` (reduced by eligibility filtering). For the default problem (7 assets, 4 obligations, 10 chunks), this is ~210 variables.

### Crossover Benchmark (MIP vs QUBO)

```
python main.py
```

Generates random problem instances of increasing size and compares MIP and QUBO on each. Prints a results table showing cost, runtime, feasibility, and winner for each configuration.

Benchmark parameters (configured at the top of `main()`):

| Parameter | Default | Description |
|---|---|---|
| `LOT_SIZE` | `1_000_000` | Lot size for generated problems |
| `MIP_TIME_LIMIT` | `10.0` | Per-problem MIP time budget in seconds |
| `QUBO_CHUNKS` | `10` | Chunks for QUBO discretisation |
| `QUBO_READS` | `3` | SA runs per problem (reduced for speed) |
| `QUBO_SWEEPS` | `4000` | SA sweeps per run |
| `QUBO_PENALTY` | `5.0` | Penalty weight for QUBO constraints |

The benchmark tests 11 configurations from 7x4 (trivial) to 50x18 (large), progressively adding MTA and concentration constraints to stress-test the MIP solver.

## Problem Data

Edit `problem_data.py` to define your own problem. Each asset requires:

```python
{"name": "US Treasury 2Y", "market_value": 50_000_000, "haircut": 0.02, "opportunity_cost": 0.005}
```

| Field | Type | Description |
|---|---|---|
| `name` | str | Display name |
| `market_value` | float | Total available market value in dollars |
| `haircut` | float | Regulatory haircut (0.0 to 1.0). Effective value = `market_value * (1 - haircut)` |
| `opportunity_cost` | float | Cost per dollar of pledging this asset (cheapest-to-deliver ordering) |

Each obligation requires:

```python
{"name": "CCP LCH - IRS Portfolio", "required_value": 35_000_000, "eligible_assets": [0, 1, 2, 4, 5]}
```

| Field | Type | Description |
|---|---|---|
| `name` | str | Display name |
| `required_value` | float | Required collateral value (after haircuts) in dollars |
| `eligible_assets` | list[int] | Indices into the assets list indicating which assets this obligation accepts |

## Solver Comparison

| Solver | Type | Variables | Constraints | Optimal? | Speed |
|---|---|---|---|---|---|
| LP | Continuous | `A * O` | Linear | Yes (global) | Fastest |
| MIP | Integer + binary | `2 * A * O` | Linear + big-M | Yes (global, if within time limit) | Exponential worst-case |
| QUBO | Binary | `A * O * K` | Penalty-based (soft) | No (heuristic) | Polynomial in problem size |

## Regulatory Context

The haircut schedules and collateral eligibility rules follow the Basel III framework (BCBS 189). The standardised supervisory haircuts from paragraph 151 are:

| Asset Type | Rating | Maturity | Haircut |
|---|---|---|---|
| Sovereign | AAA to AA | < 1 year | 0.5% |
| Sovereign | AAA to AA | 1-5 years | 2% |
| Sovereign | AAA to AA | > 5 years | 4% |
| Corporate | A to BBB | < 1 year | 1% |
| Corporate | A to BBB | 1-5 years | 3% |
| Corporate | A to BBB | > 5 years | 6% |
| Main index equities | -- | -- | 15% |
| Other equities | -- | -- | 25% |
| Cash (same currency) | -- | -- | 0% |
