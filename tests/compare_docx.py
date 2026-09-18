"""Compare the table cells of two Word documents.

Usage: python tests/compare_docx.py REFERENCE.docx NEW.docx
Prints the first differing cell of each table, or OK when every table is identical.
"""
import sys

from docx import Document


def cells(path):
    out = []
    for t in Document(path).tables:
        out.append([[c.text for c in r.cells] for r in t.rows])
    return out


def compare(ref, new):
    a, b = cells(ref), cells(new)
    ok = True
    if len(a) != len(b):
        print(f"table count {len(a)} vs {len(b)}")
        ok = False
    for k, (ta, tb) in enumerate(zip(a, b)):
        if len(ta) != len(tb):
            print(f"table {k + 1}: row count {len(ta)} vs {len(tb)}")
            ok = False
        for i, (ra, rb) in enumerate(zip(ta, tb)):
            if ra != rb:
                diff = [(x, y) for x, y in zip(ra, rb) if x != y]
                print(f"table {k + 1}, row {i}: {diff[:2]}")
                ok = False
                break
    return ok


if __name__ == "__main__":
    good = compare(sys.argv[1], sys.argv[2])
    print("OK - all table cells identical" if good else "DIFFERENT")
    sys.exit(0 if good else 1)
