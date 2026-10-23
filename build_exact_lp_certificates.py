"""Build and verify exact rational certificates for the plotted LP values.

The floating LP is used only to identify an active basis.  That basis is then
solved over fractions, and every primal and dual inequality is rechecked in
exact arithmetic before the certificate is written.
"""

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "exact_lp_certificates.json"


def kappa(n, k, j):
    numerator = sum(
        (-1)**ell * math.comb(k, ell) * math.comb(n-k, j-ell)
        for ell in range(max(0, j-(n-k)), min(j, k)+1)
    )
    return Fraction(numerator, math.comb(n, j))


def frac_text(value):
    return str(value.numerator) if value.denominator == 1 else (
        f"{value.numerator}/{value.denominator}"
    )


def solve_affine(rows, rhs, target):
    """Return the target-nearest exact solution of a rational affine system."""
    matrix = [list(row) + [value] for row, value in zip(rows, rhs)]
    nrows = len(matrix)
    ncols = len(matrix[0]) - 1
    pivots = []
    row = 0
    for col in range(ncols):
        pivot = next((r for r in range(row, nrows) if matrix[r][col]), None)
        if pivot is None:
            continue
        matrix[row], matrix[pivot] = matrix[pivot], matrix[row]
        scale = matrix[row][col]
        matrix[row] = [value / scale for value in matrix[row]]
        for r in range(nrows):
            if r != row and matrix[r][col]:
                scale = matrix[r][col]
                matrix[r] = [
                    matrix[r][c] - scale * matrix[row][c]
                    for c in range(ncols + 1)
                ]
        pivots.append(col)
        row += 1
        if row == nrows:
            break

    for r in range(nrows):
        if all(matrix[r][c] == 0 for c in range(ncols)) and matrix[r][-1]:
            raise RuntimeError("inconsistent active basis")

    free = [col for col in range(ncols) if col not in pivots]
    solution = [Fraction(0) for _ in range(ncols)]
    for col in free:
        solution[col] = Fraction(float(target[col])).limit_denominator(10**12)
    for r in range(len(pivots)-1, -1, -1):
        col = pivots[r]
        solution[col] = matrix[r][-1] - sum(
            matrix[r][free_col] * solution[free_col]
            for free_col in free
        )
    return solution


def fourier_certificate(n, m):
    kap_float = np.array([
        [float(kappa(n, k, j)) for j in range(n+1)]
        for k in range(n+1)
    ])
    objective = np.zeros(n+1)
    objective[1] = -1
    result = linprog(
        objective,
        A_ub=-kap_float[1:], b_ub=np.zeros(n),
        A_eq=np.vstack([np.eye(1, n+1, 0), np.ones(n+1)]),
        b_eq=[1, 2.0**m], bounds=(0, None), method="highs",
        options={
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
            "ipm_optimality_tolerance": 1e-12,
        },
    )
    if not result.success:
        raise RuntimeError(result.message)

    support = np.where(result.x > 1e-8)[0].tolist()
    moment_values = kap_float[1:] @ result.x
    active_moments = (np.where(abs(moment_values) < 1e-7)[0] + 1).tolist()
    active_dual = (
        np.where(abs(result.ineqlin.marginals) > 1e-10)[0] + 1
    ).tolist()

    primal_rows = [
        [Fraction(int(j == 0)) for j in support],
        [Fraction(1) for _ in support],
    ] + [
        [kappa(n, k, j) for j in support] for k in active_moments
    ]
    primal_rhs = [Fraction(1), Fraction(2**m)] + [
        Fraction(0) for _ in active_moments
    ]
    primal_active = solve_affine(
        primal_rows, primal_rhs, [result.x[j] for j in support]
    )
    b = [Fraction(0) for _ in range(n+1)]
    for j, value in zip(support, primal_active):
        b[j] = value

    dual_rows = []
    dual_rhs = []
    for j in support:
        dual_rows.append(
            [Fraction(int(j == 0)), Fraction(1)]
            + [-kappa(n, k, j) for k in active_dual]
        )
        dual_rhs.append(Fraction(int(j == 1)))
    dual_target = [
        -result.eqlin.marginals[0], -result.eqlin.marginals[1]
    ] + [
        -result.ineqlin.marginals[k-1] for k in active_dual
    ]
    dual_active = solve_affine(dual_rows, dual_rhs, dual_target)
    u, v = dual_active[:2]
    y = [Fraction(0) for _ in range(n+1)]
    for k, value in zip(active_dual, dual_active[2:]):
        y[k] = value

    certificate = {
        "coordinate": "fourier",
        "n": n,
        "m": m,
        "b": [frac_text(value) for value in b],
        "dual_u": frac_text(u),
        "dual_v": frac_text(v),
        "dual_y": [frac_text(value) for value in y],
    }
    verify(certificate)
    return certificate


def high_rate_certificate(n, m):
    gap = n - m
    if gap == 1:
        q = [Fraction(1, 2), Fraction(1, 2)] + [
            Fraction(0) for _ in range(n-1)
        ]
        u, v = Fraction(1), Fraction(-1)
        y = [Fraction(0) for _ in range(n+1)]
    elif gap == 2:
        q = [Fraction(1, 4), Fraction(1, 2), Fraction(1, 4)] + [
            Fraction(0) for _ in range(n-2)
        ]
        u, v = Fraction(3, 2), Fraction(-2)
        y = [Fraction(0) for _ in range(n+1)]
        y[n] = Fraction(1, 2)
    else:
        raise ValueError("not a high-rate endpoint")
    certificate = {
        "coordinate": "distance",
        "n": n,
        "m": m,
        "q": [frac_text(value) for value in q],
        "dual_u": frac_text(u),
        "dual_v": frac_text(v),
        "dual_y": [frac_text(value) for value in y],
    }
    verify(certificate)
    return certificate


def distance_certificate(n, m):
    """Exact certificate in q-coordinates, used when n-m is fixed."""
    alpha = Fraction(2**m, 2**n)
    kap_float = np.array([
        [float(kappa(n, k, j)) for k in range(n+1)]
        for j in range(n+1)
    ])
    result = linprog(
        np.arange(n+1, dtype=float),
        A_ub=-kap_float[1:], b_ub=np.zeros(n),
        A_eq=np.ones((1, n+1)), b_eq=[1],
        bounds=[(float(alpha), float(alpha))] + [(0, None)] * n,
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
            "ipm_optimality_tolerance": 1e-12,
        },
    )
    if not result.success:
        raise RuntimeError(result.message)

    support = np.where(result.x > 1e-8)[0].tolist()
    moment_values = kap_float[1:] @ result.x
    active_moments = (np.where(abs(moment_values) < 1e-7)[0] + 1).tolist()
    active_dual = (
        np.where(abs(result.ineqlin.marginals) > 1e-10)[0] + 1
    ).tolist()

    primal_rows = [
        [Fraction(1) for _ in support],
        [Fraction(int(k == 0)) for k in support],
    ] + [
        [kappa(n, k, j) for k in support] for j in active_moments
    ]
    primal_rhs = [Fraction(1), alpha] + [
        Fraction(0) for _ in active_moments
    ]
    primal_active = solve_affine(
        primal_rows, primal_rhs, [result.x[k] for k in support]
    )
    q = [Fraction(0) for _ in range(n+1)]
    for k, value in zip(support, primal_active):
        q[k] = value

    dual_rows = []
    dual_rhs = []
    for k in support:
        dual_rows.append(
            [Fraction(1), Fraction(int(k == 0))]
            + [kappa(n, k, j) for j in active_dual]
        )
        dual_rhs.append(Fraction(k))
    dual_target = [
        result.eqlin.marginals[0], result.upper.marginals[0]
    ] + [
        -result.ineqlin.marginals[j-1] for j in active_dual
    ]
    dual_active = solve_affine(dual_rows, dual_rhs, dual_target)
    u, v = dual_active[:2]
    y = [Fraction(0) for _ in range(n+1)]
    for j, value in zip(active_dual, dual_active[2:]):
        y[j] = value

    certificate = {
        "coordinate": "distance",
        "n": n,
        "m": m,
        "q": [frac_text(value) for value in q],
        "dual_u": frac_text(u),
        "dual_v": frac_text(v),
        "dual_y": [frac_text(value) for value in y],
    }
    verify(certificate)
    return certificate


def parse(values):
    return [Fraction(value) for value in values]


def verify(certificate):
    n, m = certificate["n"], certificate["m"]
    u = Fraction(certificate["dual_u"])
    v = Fraction(certificate["dual_v"])
    y = parse(certificate["dual_y"])
    if len(y) != n+1 or any(value < 0 for value in y[1:]):
        raise AssertionError("invalid nonnegative dual variables")

    if certificate["coordinate"] == "fourier":
        b = parse(certificate["b"])
        assert len(b) == n+1 and b[0] == 1 and sum(b) == 2**m
        assert all(value >= 0 for value in b)
        assert all(
            sum(kappa(n, k, j) * b[j] for j in range(n+1)) >= 0
            for k in range(1, n+1)
        )
        assert all(
            u * int(j == 0) + v
            - sum(y[k] * kappa(n, k, j) for k in range(1, n+1))
            >= int(j == 1)
            for j in range(n+1)
        )
        assert b[1] == u + 2**m * v
    else:
        q = parse(certificate["q"])
        alpha = Fraction(2**m, 2**n)
        assert len(q) == n+1 and q[0] == alpha and sum(q) == 1
        assert all(value >= 0 for value in q)
        assert all(
            sum(q[k] * kappa(n, k, j) for k in range(n+1)) >= 0
            for j in range(1, n+1)
        )
        assert all(
            Fraction(k) - u - v * int(k == 0)
            - sum(y[j] * kappa(n, k, j) for j in range(1, n+1)) >= 0
            for k in range(n+1)
        )
        assert sum(Fraction(k) * q[k] for k in range(n+1)) == u + v * alpha


def main():
    points = [(30, m) for m in range(1, 28)]
    points += [(n, 3) for n in range(4, 23) if n != 30]
    certificates = [fourier_certificate(n, m) for n, m in points]
    certificates += [distance_certificate(m+5, m) for m in range(1, 51)]
    certificates += [high_rate_certificate(30, 28), high_rate_certificate(30, 29)]
    certificates.sort(key=lambda item: (item["n"], item["m"]))
    OUTPUT.write_text(json.dumps(certificates, indent=2) + "\n")
    for certificate in certificates:
        verify(certificate)
    print(f"verified {len(certificates)} exact primal-dual certificates")


if __name__ == "__main__":
    main()
