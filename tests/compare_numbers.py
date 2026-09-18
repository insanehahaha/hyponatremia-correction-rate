"""Compare the numbers in two Excel sheets, ignoring labels and column order.

Usage: python tests/compare_numbers.py REFERENCE.xlsx SHEET NEW.xlsx SHEET
All numbers are extracted row by row, skipping the label column (from text cells too, e.g. "17.7 (13.7-22.0)") and
compared in order; a mismatch prints the first differing row.
"""
import re
import sys

import pandas as pd

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def numbers(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    rows = []
    for _, r in df.iterrows():
        vals = []
        for v in r.values[1:]:          # first column holds the row label
            if pd.isna(v):
                continue
            vals += [float(x) for x in NUM.findall(str(v).replace("–", "-").replace("−", "-"))]
        rows.append(vals)
    return rows


def compare(ref, ref_sheet, new, new_sheet, tol=1e-9):
    a, b = numbers(ref, ref_sheet), numbers(new, new_sheet)
    if len(a) != len(b):
        return False, f"row count {len(a)} vs {len(b)}"
    for i, (ra, rb) in enumerate(zip(a, b)):
        if len(ra) != len(rb) or any(abs(x - y) > tol for x, y in zip(ra, rb)):
            return False, f"row {i}: {ra} vs {rb}"
    return True, f"{len(a)} rows identical"


if __name__ == "__main__":
    ok, msg = compare(*sys.argv[1:5])
    print("OK" if ok else "DIFFERENT", "-", msg)
    sys.exit(0 if ok else 1)
