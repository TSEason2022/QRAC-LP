# QRAC-LP

Core computation and exact rational certification for a linear-programming converse on binary quantum random access codes (QRACs). An input pair `(n, m)` represents `n` classical bits encoded into `m` qubits, with `1 <= m < n`. The output `B_LP` is an upper bound on the optimal average bit-recovery probability.

## Core design

- **Normalized Krawtchouk recurrence:** builds the radial constraint matrix without repeatedly evaluating binomial sums in the floating solver.
- **Two equivalent coordinates:** Fourier variables `b_j` avoid a tiny `q_0 = 2^(m-n)` coefficient; distance variables `q_k` are useful when the redundancy `n-m` is fixed and small. The CLI lets the user select the coordinate system.
- **HiGHS optimization:** SciPy solves an LP with `n+1` variables. The original implementation handles the two known high-rate endpoints analytically in its Fourier entry point.
- **Exact certification:** a floating solve identifies an active basis; `Fraction` arithmetic reconstructs and checks primal feasibility, dual feasibility, and equality of the objectives. A separate routine certifies the minimizing integer order in the Akibue moment bound.

These choices provide a compact, numerically conditioned implementation. No timing benchmark or speedup factor is claimed.

## Quick start

Python 3.12 and [uv](https://docs.astral.sh/uv/) are used for the pinned environment. From this directory:

```bash
uv sync --locked
uv run --locked python qrac_lp.py bounds 30 25 --coordinates distance --compare
uv run --locked python qrac_lp.py certificate 30 25 --coordinates distance --output result.json
uv run --locked python qrac_lp.py verify result.json
uv run --locked python qrac_lp.py moment-order 30 25
```

To check the supplied certificate without running an optimizer:

```bash
uv run --locked python qrac_lp.py verify examples/n30_m25_certificate.json
```

Commands emit JSON to stdout unless `--output PATH` is supplied. `bounds` returns `B_LP`, the LP distortion `delta_LP`, solver status, and tolerances. `--compare` adds the Akibue, Nayak, and Mancinska–Storgaard bounds. `certificate` writes a single exact rational primal-dual witness; `verify` checks it without solving an LP. `moment-order` returns the exact minimizing integer order and the number of orders checked. No random seed is used.

Run `python qrac_lp.py --help` for the command list. Keep Python assertions enabled: do not use `python -O`. The CLI rejects optimized mode because the original certificate checker uses assertions.

## Files and evidence

| File | Purpose |
| --- | --- |
| `reproduce_analytic_bounds.py` | Original floating LP, analytic-bound evaluations, and CSV reproduction routines. |
| `build_exact_lp_certificates.py` | Original exact rational primal-dual certificate builder and checker. |
| `certify_akibue_orders.py` | Original exact integer/rational moment-order certification. |
| `qrac_lp.py` | Added portable CLI for one parameter pair at a time. |
| `examples/n30_m25_certificate.json` | One existing exact distance-coordinate certificate; its LP objective is `15/8`. |
| `examples/data_fixed_n30.csv` | Existing computed comparisons for `n=30`, `1 <= m < 30`; includes all four bound columns and the selected moment order. |

The three original Python sources are unchanged. The CLI and packaging files were added; the example certificate was selected from the existing certificate collection. The CSV was copied unchanged from a saved numerical-reproduction output, rather than the manuscript's source tables, and may differ from those tables in the final floating-point digits. The public repository normalizes CSV line endings to LF; its numerical content is unchanged. No manuscript, figures, private paths, or temporary experiment logs are included.

**Numerical results:** `bounds` and the saved CSV contain floating evaluations. For example, the saved `n=30` slice has `B_LP` below the Nayak, Mancinska–Storgaard, and Akibue columns at each point `6 <= m < 30`. This is a finite comparison range; neither the CSV nor the CLI establishes universal dominance over other converses.

**Certified results:** a successfully checked exact primal-dual certificate establishes the optimum of this LP at its stated `(n,m)`. Decimal square roots displayed by the CLI are numerical. LP optimality does not by itself establish that a QRAC attains the bound. The single included certificate does not certify every CSV row; full certificate collections are outside this core package.

The original scripts retain their reference batch entry points. Directly executing `build_exact_lp_certificates.py` or `certify_akibue_orders.py` runs their full reference sweeps and writes JSON beside the script; directly executing `reproduce_analytic_bounds.py` evaluates multiple parameter slices (`--output-dir` optionally saves CSVs). Use `qrac_lp.py` for single-point work. Large inputs and exact rational reconstruction can be expensive or encounter floating conditioning limits.

## Validation

Package checks used Python 3.12.13, NumPy 2.4.6, and SciPy 1.17.1. Syntax/import checks, a small `(6,3)` floating LP with analytic comparisons, single-point exact certificate generation, moment-order certification, and verification of the supplied `(30,25)` certificate passed. The small floating LP reported success with primal/dual feasibility tolerances `1e-9` and IPM optimality tolerance `1e-12`; certificate generation uses `1e-10`, `1e-10`, and `1e-12` only for active-basis discovery. Final rational checks use tolerance zero and no solver. No full reference sweep or timing benchmark was run during packaging.

## Research reference

Shuo Tan and Syed A. Jafar. **Optimal Average Success Probabilities of Binary (n, n-1) and (n, n-2) Quantum Random Access Codes.** Long version, UC Irvine: Recent Work, October 21, 2026.

- [Paper and abstract](https://tseason2022.github.io/papers/qrac/)
- [Full-text PDF](https://tseason2022.github.io/papers/qrac/qrac-long-version.pdf)
- [Institutional archive on eScholarship](https://escholarship.org/uc/item/5910g713)
