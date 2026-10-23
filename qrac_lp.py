"""Single-point command-line interface for the accompanying QRAC LP sources."""

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path


def parameter_pair(n, m):
    if type(n) is not int or type(m) is not int or not (1 <= m < n):
        raise ValueError("Expected integers with 1 <= m < n.")


def add_parameters(parser):
    parser.add_argument("n", type=int, help="number of classical input bits")
    parser.add_argument("m", type=int, help="number of encoding qubits; 1 <= m < n")


def add_coordinates(parser):
    parser.add_argument(
        "--coordinates", choices=("fourier", "distance"), default="fourier",
        help="Fourier b variables, or distance q variables (useful for a fixed small n-m)",
    )


def add_output(parser):
    parser.add_argument("--output", type=Path, help="write JSON to this path instead of stdout")


def bounds(args):
    from reproduce_analytic_bounds import (
        akibue_bound, lp_bound, lp_bound_q, ms_bound, nayak_bound,
    )

    solver = lp_bound if args.coordinates == "fourier" else lp_bound_q
    upper, delta, vector = solver(args.n, args.m)
    analytic = args.coordinates == "fourier" and vector is None
    result = {
        "n": args.n,
        "m": args.m,
        "evidence": "floating-point evaluation; use an exact certificate for LP optimality",
        "coordinates": args.coordinates,
        "B_LP": upper,
        "delta_LP": delta,
        "solver": None if analytic else "SciPy linprog / HiGHS",
        "solver_status": "analytic high-rate endpoint" if analytic else "success",
        "seed": None,
        "solver_tolerances": None if analytic else {
            "primal_feasibility": 1e-9,
            "dual_feasibility": 1e-9,
            "ipm_optimality": 1e-12,
        },
    }
    if args.compare:
        akibue, ell = akibue_bound(args.n, args.m)
        result["comparison"] = {
            "B_Akibue": akibue,
            "B_Nayak": nayak_bound(args.n, args.m),
            "B_MS": ms_bound(args.n, args.m),
            "ell_star_numerical": ell,
            "evidence": "numerical comparison; moment-order provides an exact integer-order certificate",
        }
    return result


def certificate(args):
    from build_exact_lp_certificates import (
        distance_certificate, fourier_certificate, high_rate_certificate,
    )

    if args.n - args.m in (1, 2):
        return high_rate_certificate(args.n, args.m)
    builder = fourier_certificate if args.coordinates == "fourier" else distance_certificate
    return builder(args.n, args.m)


def verify_certificate(args):
    from build_exact_lp_certificates import verify

    record = json.loads(args.certificate.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("coordinate") not in ("fourier", "distance"):
        raise ValueError("Expected one Fourier or distance certificate JSON object.")
    n, m = record["n"], record["m"]
    parameter_pair(n, m)
    verify(record)
    if record["coordinate"] == "fourier":
        delta = (Fraction(n) - Fraction(record["b"][1])) / 2
    else:
        delta = sum(Fraction(k) * Fraction(q) for k, q in enumerate(record["q"]))
    return {
        "n": n,
        "m": m,
        "coordinate": record["coordinate"],
        "status": "exact primal-dual checks passed",
        "delta_LP_exact": str(delta),
        "B_LP_numerical": (1 + math.sqrt(1 - 2 * float(delta) / n)) / 2,
        "evidence": "exact rational LP certificate; the displayed square root is numerical",
        "solver": None,
        "seed": None,
        "tolerance": 0,
    }


def moment_order(args):
    from certify_akibue_orders import certify

    return {**certify(args.n, args.m),
            "evidence": "exact integer/rational comparison of all candidate and future orders",
            "seed": None, "tolerance": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate = commands.add_parser("bounds", help="evaluate one LP converse; optionally compare analytic bounds")
    add_parameters(evaluate)
    add_coordinates(evaluate)
    add_output(evaluate)
    evaluate.add_argument("--compare", action="store_true", help="also evaluate Akibue, Nayak, and MS bounds")
    evaluate.set_defaults(run=bounds)

    construct = commands.add_parser("certificate", help="construct and check one exact rational LP certificate")
    add_parameters(construct)
    add_coordinates(construct)
    add_output(construct)
    construct.set_defaults(run=certificate)

    verify = commands.add_parser("verify", help="check an existing single-certificate JSON file")
    verify.add_argument("certificate", type=Path)
    add_output(verify)
    verify.set_defaults(run=verify_certificate)

    moment = commands.add_parser("moment-order", help="certify the minimizing integer moment order at one point")
    add_parameters(moment)
    add_output(moment)
    moment.set_defaults(run=moment_order)

    args = parser.parse_args(argv)
    if not __debug__:
        parser.error("Run without -O: the original certificate verifier uses Python assertions.")
    try:
        if args.command != "verify":
            parameter_pair(args.n, args.m)
        result = args.run(args)
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        else:
            print(serialized, end="")
    except (ValueError, RuntimeError, AssertionError, KeyError, OSError) as error:
        parser.exit(1, f"error: {error or type(error).__name__}\n")


if __name__ == "__main__":
    main()
