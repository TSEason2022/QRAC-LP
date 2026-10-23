"""Reproduce the LP and analytic converse curves in the QRAC draft.

The LP is solved in the Fourier variables b_j = binom(n,j) a_j.  This
coordinate system avoids the tiny q_0=2^{m-n} coefficient in the radial
distance-distribution formulation.
"""

import argparse
import csv
import math
from pathlib import Path
import numpy as np
from scipy.optimize import brentq, linprog
from scipy.special import gammaln, logsumexp


def normalized_krawtchouk(n):
    """Matrix kap[j,k] = K_j^(n)(k)/binom(n,j), using a recurrence."""
    k = np.arange(n + 1, dtype=float)
    K = np.empty((n + 1, n + 1), dtype=float)
    K[0] = 1.0
    K[1] = n - 2*k
    for j in range(1, n):
        K[j+1] = ((n-2*k)*K[j] - (n-j+1)*K[j-1])/(j+1)
    K /= np.array([math.comb(n,j) for j in range(n+1)])[:, None]
    return K


def lp_bound(n, m):
    """Return (B_LP, delta_LP, b) for a binary (n,m) QRAC."""
    # The two high-rate values are known exactly from the analytic proof.
    # Handling them explicitly also avoids asking a floating solver to
    # distinguish constraints at scales of order 2**(m-n).
    if m in (n-1, n-2):
        delta = (n-m)/2
        bound = (1 + math.sqrt(m/n))/2
        return bound, delta, None
    kap = normalized_krawtchouk(n)
    # Maximize b_1 subject to b>=0, b_0=1, sum_j b_j=2^m, and
    # sum_j b_j kap[j,k]>=0.  This floating version is suitable for the
    # plotted fixed-n (m<=29) and fixed-m=5 slices.  Exact-rational LP
    # certificates should be used for final publication numbers.
    c = np.zeros(n+1); c[1] = -1
    D = math.ldexp(1.0,m)
    result = linprog(
        c, A_ub=-kap.T, b_ub=np.zeros(n+1),
        A_eq=np.ones((1,n+1)), b_eq=[D],
        bounds=[(1,1)] + [(0,None)]*n,
        method="highs",
        options={"primal_feasibility_tolerance": 1e-9,
                 "dual_feasibility_tolerance": 1e-9,
                 "ipm_optimality_tolerance": 1e-12})
    if not result.success:
        raise RuntimeError(result.message)
    b = result.x
    delta = (n-b[1])/2
    bound = (1 + math.sqrt(max(0, 1-2*delta/n)))/2
    # QA diagnostics.  For publication, save exact-rational primal/dual
    # certificates; these floats are intended only to reproduce curves.
    assert abs(b.sum()-D) < 1e-6*max(1,D)
    assert (kap.T@b).min() > -2e-7*max(1,D)
    return bound, delta, b


def lp_bound_q(n, m):
    """Original q_k LP; best-conditioned when n-m is fixed/moderate."""
    kap = normalized_krawtchouk(n)
    result = linprog(
        np.arange(n+1,dtype=float),
        A_ub=-kap[1:], b_ub=np.zeros(n),
        A_eq=np.ones((1,n+1)), b_eq=[1.0],
        bounds=[(math.ldexp(1.0,m-n),)*2]+[(0,None)]*n,
        method="highs",
        options={"primal_feasibility_tolerance":1e-9,
                 "dual_feasibility_tolerance":1e-9,
                 "ipm_optimality_tolerance":1e-12})
    if not result.success: raise RuntimeError(result.message)
    delta=result.fun
    bound=(1+math.sqrt(max(0,1-2*delta/n)))/2
    return bound,delta,result.x


def log_even_moment(n, ell):
    k = np.arange(n+1)
    z = np.abs(n-2*k); nz = z > 0
    log_prob = (gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)
                - n*math.log(2))
    return logsumexp(log_prob[nz] + 2*ell*np.log(z[nz]))


def akibue_bound(n, m, ell_cap=10000):
    best = (math.inf, None)
    best_root = math.inf
    for ell in range(1,ell_cap+1):
        root = math.exp(((m-1)*math.log(2)+log_even_moment(n,ell))/(2*ell))
        value = (1+root/n)/2
        if value < best[0]: best,best_root = (value,ell),root
        # The k=0,n terms alone imply, for every r>=ell+1,
        # F_r^(1/(2r)) >= n*2^((m-n)/(2r))
        #                   >= n*2^((m-n)/(2(ell+1))).
        # Hence this is a global (not local/unimodality-based) stopping test.
        future_lower = n*2**((m-n)/(2*(ell+1)))
        if future_lower >= best_root*(1+2e-14): break
    else: raise RuntimeError("ell cap too small")
    return best


def ms_bound(n,m):
    return min(1,(1+math.sqrt(2**(m-1)/n))/2)


def nayak_bound(n,m):
    target=1-m/n
    if target == 1: return .5
    h2=lambda p: -p*math.log2(p)-(1-p)*math.log2(1-p)
    return brentq(lambda p:h2(p)-target,.5,1-1e-15)


def rows(points, lp_solver=lp_bound):
    out=[]
    for n,m in points:
        blp,delta,_=lp_solver(n,m)
        ba,ell=akibue_bound(n,m)
        out.append((n,m,blp,ba,nayak_bound(n,m),ms_bound(n,m),ell,delta))
    return out


def closed_family_rows(r_max=10):
    """Values for the two closed-form families in Theorem 4."""
    out = []
    for family, r_min in (("A", 2), ("B", 3)):
        for r in range(r_min, r_max + 1):
            n = 2**r - (1 if family == "A" else 2)
            m = n - r
            if family == "A":
                lp = (1 + math.sqrt((n-1)/(n+3))) / 2
            else:
                lp = (1 + math.sqrt((n-2)/(n+2))) / 2
            akibue, ell = akibue_bound(n, m)
            out.append((family, r, n, m, lp, akibue, akibue-lp, ell))
    return out


def write_family_csv(path, data):
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "family", "r", "n", "m", "B_LP_closed", "B_Akibue",
            "B_Akibue_minus_B_LP", "ell_star",
        ])
        for family, r, n, m, lp, akibue, difference, ell in data:
            writer.writerow([
                family, r, n, m, f"{lp:.15g}", f"{akibue:.15g}",
                f"{difference:.15g}", ell,
            ])


def write_csv(path, data):
    """Write the values used by the manuscript figures."""
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "n", "m", "B_LP", "B_Akibue", "B_Nayak", "B_MS",
            "ell_star", "delta_LP",
        ])
        for row in data:
            writer.writerow([
                row[0], row[1],
                *(f"{value:.15g}" for value in row[2:6]),
                row[6], f"{row[7]:.15g}",
            ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        help="optional directory for one CSV file per parameter slice",
    )
    args = parser.parse_args()
    slices = {
        "fixed_n30": [(30,m) for m in range(1,30)],
        "fixed_m5": [(n,5) for n in range(6,61)],
        "gap5": [(m+5,m) for m in range(1,51)],
        "fixed_m3": [(n,3) for n in range(4,23)],
    }
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for name,points in slices.items():
        data=rows(points, lp_bound_q if name=="gap5" else lp_bound)
        if args.output_dir:
            write_csv(args.output_dir / f"{name}.csv", data)
        print("\n",name)
        for r in data: print(*r)
    family_data = closed_family_rows()
    if args.output_dir:
        write_family_csv(args.output_dir / "closed_families.csv", family_data)
    print("\n closed_families")
    for item in family_data:
        print(*item)
