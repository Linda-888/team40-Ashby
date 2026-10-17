"""Step 11 helper: collect every model's results into one privacy scorecard (Markdown + CSV).

    python 08_scorecard.py            # after 05/06/07 ran for each model
"""
import json

import pandas as pd

from fingerprints import paths


def load(model, name):
    f = paths.root() / "results" / model / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def rate(s):
    if not s or not s.get("n_queries"):
        return "n/a"
    return f"{s['top1']:.1%} vs {s['chance_top1']:.1%} chance (n={s['n_queries']})"


def to_markdown(df):
    """Plain Markdown table (avoids needing the tabulate package)."""
    cols = ["", *df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join([str(i), *map(str, r)]) + " |" for i, r in zip(df.index, df.values)]
    return "\n".join(lines)


def main():
    rows = []
    for m in paths.MODELS:
        t, sc, rm = load(m, "tests"), load(m, "scramble"), load(m, "removal")
        if not t:
            continue
        row = {
            "model": m,
            "slide-to-slide (all)": rate(t.get("test1", {}).get("all")),
            "slide-to-slide (same site)": rate(t.get("test1", {}).get("site")),
            "report-to-slide (all)": rate(t.get("test2", {}).get("all")),
            "report-to-slide (same dx + site) HEADLINE": rate(t.get("test2", {}).get("dx+site")),
        }
        for name in ("test1", "test2"):
            r = sc.get("tests", {}).get(name)
            if r:
                broken = {a: k for a, k in r["k_to_half"].items() if k is not None}
                if broken:
                    a = min(broken, key=broken.get)
                    row[f"key scrambling, {name}"] = f"broken with {broken[a]} known pairs ({a})"
                else:
                    row[f"key scrambling, {name}"] = f"survives up to k={max(r['ks'])}"
        for method in ("noise", "remove"):
            res = rm.get("methods", {}).get(method)
            if res:
                s = res["sweet_spot"]
                row[f"fix: {method}"] = (f"strength {s['strength']:g}, accuracy {res['baseline_utility']:.3f}"
                                         f" -> {s['utility']:.3f}" if s else "no setting blocks both tests")
        rows.append(row)

    if not rows:
        print("no results yet: run 05_tests.py (and 06, 07) first")
        return
    df = pd.DataFrame(rows).set_index("model").T
    out = paths.root() / "results"
    df.to_csv(out / "scorecard.csv")
    md = "# Privacy scorecard\n\n" + to_markdown(df) + "\n"
    (out / "scorecard.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
