"""Steps 6-7: Test 1 (slide to slide) and Test 2 (report to slide), with every fairness control.

    python 05_tests.py --model conch
Writes results/<model>/tests.json. The headline is Test 2 restricted to same diagnosis + same site.
"""
import argparse
import json

from fingerprints import data, paths
from fingerprints.matching import CONTROLS, fmt, test1, test2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=paths.MODELS)
    ap.add_argument("--reports-name", default="reports")
    args = ap.parse_args()

    e = data.load(args.model, args.reports_name)
    out = {"model": args.model, "n_patients": e.n, "test1": {}, "test2": {}}
    for name, fn, has in (("test1", test1, e.has_B), ("test2", test2, e.has_R)):
        if has.sum() < 2:
            print(f"{name}: not enough patients, skipped")
            continue
        print(f"\n{name} ({'slide A -> slide B' if name == 'test1' else 'report -> slide A'})")
        for control in CONTROLS:
            s = fn(e, control)
            out[name][control] = s
            print(f"  {control:8s} {fmt(s)}")

    if out["test2"].get("dx+site"):
        print(f"\nHEADLINE (Test 2, same diagnosis + same site): {fmt(out['test2']['dx+site'])}")
    dst = paths.results_dir(args.model) / "tests.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
