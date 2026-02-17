# Collateral Optimisation

Solves the bank collateral allocation problem using four approaches: Linear Programming (LP), Mixed-Integer Programming (MIP), Quadratic Unconstrained Binary Optimisation (QUBO), and Constrained Quadratic Model (CQM) with D-Wave Ocean SDK. Supports local simulated annealing, D-Wave QPU (quantum annealer), and D-Wave hybrid classical-quantum solvers. Includes a crossover benchmark comparing MIP vs QUBO vs CQM at scale.

## Problem

A bank holds a portfolio of assets (bonds, cash, equities) that must be posted as collateral against multiple margin obligations (CCP trades, bilateral CSAs, repos). Each obligation has a required value and a set of eligible asset types. Each asset has a market value, a regulatory haircut, and an opportunity cost. The goal is to minimise total opportunity cost while satisfying all obligations and never exceeding available inventory.

## Requirements

- Python 3.8+
- NumPy
- SciPy
- D-Wave Ocean SDK (`dwave-neal`, `dimod`)
- D-Wave System (optional, for QPU/hybrid backends): `dwave-system`

```
pip install numpy scipy dwave-neal dimod
```

For D-Wave QPU or hybrid cloud solvers, also install:

```
pip install dwave-system
dwave config create
```

The `dwave config create` command will prompt you for your D-Wave Leap API token. You can get a free account at [cloud.dwavesys.com](https://cloud.dwavesys.com).

The QUBO solver uses D-Wave's `neal.SimulatedAnnealingSampler` (local, no account needed), `DWaveSampler` (QPU quantum annealer), or `LeapHybridSampler` (classical-quantum hybrid). The model is built using `dimod.BinaryQuadraticModel`, which is the standard format across all D-Wave solvers.

## Project Structure

| File | Description |
|---|---|
| `problem_data.py` | Shared asset inventory and obligation definitions |
| `collateral_optimisation.py` | LP solver using `scipy.optimize.linprog` (HiGHS) |
| `collateral_mip.py` | MIP solver using `scipy.optimize.milp` (HiGHS branch-and-bound) |
| `collateral_qubo.py` | QUBO solver using D-Wave Ocean SDK (`dimod` + `neal` / QPU / hybrid) |
| `collateral_cqm.py` | CQM solver using D-Wave Ocean SDK (`dimod.ConstrainedQuadraticModel`) |
| `main.py` | Crossover benchmark: MIP vs QUBO vs CQM at increasing problem sizes |
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

### QUBO Solver (D-Wave Ocean SDK)

Reformulates the problem into a `dimod.BinaryQuadraticModel` (QUBO) and solves using one of three D-Wave backends. Supports command-line arguments:

```
python collateral_qubo.py [--backend {neal,qpu,hybrid}] [options]
```

**Backends:**

| Backend | Command | Description | Requirements |
|---|---|---|---|
| `neal` (default) | `python collateral_qubo.py` | Local simulated annealing (C++ via D-Wave Neal) | `dwave-neal`, `dimod` |
| `qpu` | `python collateral_qubo.py --backend qpu` | D-Wave Advantage quantum annealer | `dwave-system` + Leap API token |
| `hybrid` | `python collateral_qubo.py --backend hybrid` | D-Wave hybrid classical-quantum solver | `dwave-system` + Leap API token |

**Common parameters:**

| CLI flag | Parameter | Default | Description |
|---|---|---|---|
| `--num-chunks` | `num_chunks` | `10` | Binary bits per (asset, obligation) pair. Each chunk represents `market_value / num_chunks` dollars. Higher = better precision but larger search space. |
| `--penalty-weight` | `penalty_weight` | `1.0` | Multiplier for constraint-violation penalties. Higher values force feasibility at the expense of objective quality. |
| `--num-reads` | `num_reads` | `20` | Number of independent SA/QPU runs. Best solution across all runs is returned. (neal and qpu only) |

**Neal-only parameters (simulated annealing):**

| CLI flag | Parameter | Default | Description |
|---|---|---|---|
| `--num-sweeps` | `num_sweeps` | `5000` | Sweeps per SA run. Each sweep visits every variable once. More sweeps allow better convergence. |
| — | `seed` | `42` | Random seed for reproducibility (Python API only). |
| — | `beta_range` | `None` | Inverse temperature range `(beta_min, beta_max)`. `None` lets Neal auto-calculate from QUBO coefficients. (Python API only) |
| — | `beta_schedule_type` | `"geometric"` | Temperature schedule: `"geometric"` or `"linear"`. Geometric cools faster at high temperatures. (Python API only) |

**QPU-only parameters (D-Wave Advantage quantum annealer):**

| CLI flag | Parameter | Default | Description |
|---|---|---|---|
| `--annealing-time` | `annealing_time` | `None` (20μs) | Annealing time in microseconds. Range: 1–2000μs. Longer times can improve solution quality. |
| `--chain-strength` | `chain_strength` | `None` (auto) | Coupling strength for embedding chains. `None` lets `EmbeddingComposite` auto-calculate. |

**Hybrid-only parameters (LeapHybridSampler):**

| CLI flag | Parameter | Default | Description |
|---|---|---|---|
| `--time-limit` | `time_limit` | `None` (auto) | Time limit in seconds (minimum 3). `None` lets the hybrid solver choose automatically. |

**Examples:**

```bash
# Local simulated annealing (default, no cloud account needed)
python collateral_qubo.py

# More precision with 20 chunks and higher penalty
python collateral_qubo.py --num-chunks 20 --penalty-weight 2.0 --num-sweeps 10000

# D-Wave QPU with 100 reads and 200μs annealing time
python collateral_qubo.py --backend qpu --num-reads 100 --annealing-time 200

# D-Wave hybrid solver with 10 second time limit
python collateral_qubo.py --backend hybrid --time-limit 10
```

**QUBO variable count**: For `A` assets, `O` obligations, and `K` chunks, the number of binary variables is up to `A * O * K` (reduced by eligibility filtering). For the default problem (7 assets, 4 obligations, 10 chunks), this is ~210 variables.

**D-Wave QPU notes**: The D-Wave Advantage QPU has ~5000 qubits. After minor embedding overhead, problems up to ~150 logical variables typically embed well. For larger problems, use the hybrid backend which handles arbitrary sizes. QPU access requires a D-Wave Leap account — configure with `dwave config create`.

### CQM Solver (Constrained Quadratic Model)

Uses `dimod.ConstrainedQuadraticModel` which encodes constraints natively (hard constraints) rather than as penalty terms. This is the key advantage over QUBO — constraints are guaranteed to be satisfied when using the hybrid solver.

```
python collateral_cqm.py [--backend {neal,hybrid}] [options]
```

**Backends:**

| Backend | Command | Description | Requirements |
|---|---|---|---|
| `neal` (default) | `python collateral_cqm.py` | CQM -> BQM conversion, solved with Neal SA | `dwave-neal`, `dimod` |
| `hybrid` | `python collateral_cqm.py --backend hybrid` | LeapHybridCQMSampler (native CQM, continuous variables) | `dwave-system` + Leap API token |

**Parameters:**

| CLI flag | Parameter | Default | Description |
|---|---|---|---|
| `--lot-size` | `lot_size` | `1_000_000` | Lot size for integer discretisation (neal only). Hybrid uses continuous Real variables. |
| `--num-reads` | `num_reads` | `20` | Number of SA reads (neal only). |
| `--num-sweeps` | `num_sweeps` | `5000` | Sweeps per SA run (neal only). |
| `--time-limit` | `time_limit` | `None` (auto) | Time limit in seconds (hybrid only, minimum 5). |
| `--lagrange` | `lagrange_multiplier` | `None` (auto) | Penalty weight for CQM-to-BQM conversion (neal only). |

**Examples:**

```bash
# Local simulated annealing (CQM -> BQM -> Neal SA)
python collateral_cqm.py

# D-Wave hybrid CQM solver (continuous variables, hard constraints)
python collateral_cqm.py --backend hybrid --time-limit 10
```

**QUBO vs CQM:** QUBO encodes all constraints as quadratic penalty terms in the objective (soft constraints). CQM keeps constraints separate from the objective. When using the hybrid backend, `LeapHybridCQMSampler` handles constraints natively — solutions are guaranteed feasible. When using the neal backend, `dimod.cqm_to_bqm()` converts constraints to penalties (similar to QUBO), so the local SA path has comparable feasibility characteristics to QUBO.

### Crossover Benchmark (MIP vs QUBO vs CQM)

```
python main.py
```

Generates random problem instances of increasing size and compares MIP, QUBO, and CQM on each. Prints a results table showing cost, runtime, feasibility, and winner for each configuration.

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
| QUBO (Neal SA) | Binary | `A * O * K` | Penalty-based (soft) | No (heuristic) | Polynomial in problem size |
| QUBO (D-Wave QPU) | Binary | `A * O * K` | Penalty-based (soft) | No (heuristic) | ~μs per anneal, constant time |
| QUBO (Hybrid) | Binary | `A * O * K` | Penalty-based (soft) | No (heuristic) | Handles large problems |
| CQM (Neal SA) | Integer | `A * O` | Hard (via BQM penalties) | No (heuristic) | Polynomial in problem size |
| CQM (Hybrid) | Continuous | `A * O` | Hard (native) | No (heuristic) | Handles large problems |

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
