"""Certify the minimizing moment order in the two manuscript figures."""

import json
import math
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def moment_value(n, m, ell):
    total = sum(
        math.comb(n, k) * abs(n-2*k)**(2*ell)
        for k in range(n+1)
    )
    return Fraction(2**m * total, 2**(n+1))


def no_larger_root(left_value, left_order, right_value, right_order):
    """Test left_value^(1/(2 left_order)) <= the corresponding right root."""
    return left_value**right_order <= right_value**left_order


def certify(n, m):
    best_order = 1
    best_value = moment_value(n, m, best_order)
    order = 1
    while True:
        value = moment_value(n, m, order)
        if no_larger_root(value, order, best_value, best_order):
            best_order, best_value = order, value

        # The k=0,n terms imply F_r >= 2^(m-n) n^(2r).
        # Since m<n, the resulting root lower bound is increasing in r.
        future_order = order + 1
        future_lower = Fraction(
            n**(2*future_order), 2**(n-m)
        )
        if no_larger_root(best_value, best_order, future_lower, future_order):
            return {
                "n": n,
                "m": m,
                "ell_star": best_order,
                "orders_checked_through": order,
            }
        order += 1


def main():
    points = [(30, m) for m in range(1, 30)]
    points += [(n, 3) for n in range(4, 23)]
    points += [(m+5, m) for m in range(1, 51)]
    points = list(dict.fromkeys(points))
    certificates = [certify(n, m) for n, m in points]
    output = ROOT / "akibue_order_certificates.json"
    output.write_text(json.dumps(certificates, indent=2) + "\n")
    print(
        f"certified {len(certificates)} moment minimizations; "
        f"largest ell* = {max(item['ell_star'] for item in certificates)}"
    )


if __name__ == "__main__":
    main()
