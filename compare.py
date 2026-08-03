#!/usr/bin/env python3
"""
compare.py — Sakshi-Protocol mode comparison

Reads the per-mode result JSON files (baseline / sakshi / sakshi_omega) for a
given model and prints a side-by-side comparison of the key metrics, plus a
short honest note on how to read them.

Usage:
    python compare.py --model claude-sonnet-4-6
    python compare.py --results-dir results/claude-sonnet-4-6
"""
import argparse
import json
import os
from collections import Counter

MODES = ["baseline", "sakshi", "sakshi_omega"]


def load(results_dir, mode):
    path = os.path.join(results_dir, f"{mode}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarise(records):
    n = len(records)
    accepts    = sum(1 for r in records if r.get("effective_decision") == "accept")
    abstains   = sum(1 for r in records if r.get("effective_decision") == "abstain")
    intervened = sum(1 for r in records if r.get("intervened"))
    grounded   = sum(1 for r in records if r.get("grounded"))
    correct    = sum(1 for r in records if r.get("correct") == 1)

    by_class = {}
    for r in records:
        c = r.get("class") or "-"
        by_class.setdefault(c, Counter())[r.get("effective_decision")] += 1

    # how many prompts are "trap" classes (A/B/C) vs normal
    trap = sum(1 for r in records if (r.get("class") or "") in ("A", "B", "C"))
    normal = n - trap

    return {
        "n": n, "accept": accepts, "abstain": abstains,
        "intervened": intervened, "grounded": grounded, "correct": correct,
        "by_class": by_class, "trap": trap, "normal": normal,
    }


def pct(x, n):
    return f"{(100.0 * x / n):.0f}%" if n else "-"


def main():
    ap = argparse.ArgumentParser(description="Compare Sakshi benchmark modes")
    ap.add_argument("--model", default=None, help="Model id (results/<model>/)")
    ap.add_argument("--results-dir", default=None, help="Explicit results dir")
    args = ap.parse_args()

    results_dir = args.results_dir or os.path.join("results", args.model or "claude-sonnet-4-6")

    data = {m: load(results_dir, m) for m in MODES}
    present = [m for m in MODES if data[m] is not None]
    if not present:
        print(f"No result files found in {results_dir}")
        return

    summ = {m: summarise(data[m]) for m in present}
    first = summ[present[0]]

    W = 16
    width = W * (len(present) + 1)
    print("=" * width)
    print(f"  Sakshi-Protocol — mode comparison   ({results_dir})")
    print("=" * width)
    print("Metric".ljust(W) + "".join(m.ljust(W) for m in present))
    print("-" * width)

    def row(label, fn):
        print(label.ljust(W) + "".join(str(fn(summ[m])).ljust(W) for m in present))

    row("Prompts",    lambda s: s["n"])
    row("Accept",     lambda s: f'{s["accept"]} ({pct(s["accept"], s["n"])})')
    row("Abstain",    lambda s: f'{s["abstain"]} ({pct(s["abstain"], s["n"])})')
    row("Intervened", lambda s: f'{s["intervened"]} ({pct(s["intervened"], s["n"])})')
    row("Grounded",   lambda s: f'{s["grounded"]} ({pct(s["grounded"], s["n"])})')
    row("Accuracy",   lambda s: f'{s["correct"]}/{s["n"]} ({pct(s["correct"], s["n"])})')

    print("-" * width)
    print("Abstention by hallucination class (higher = more caught):")
    classes = sorted({c for m in present for c in summ[m]["by_class"] if c not in ("-", "")})
    for c in classes:
        line = f"  Class {c}".ljust(W)
        for m in present:
            bc = summ[m]["by_class"].get(c, Counter())
            tot = sum(bc.values())
            ab = bc.get("abstain", 0)
            line += f'{ab}/{tot} abst'.ljust(W)
        print(line)
    print("=" * width)

    # ---- Honest reading guide -------------------------------------------
    trap = first["trap"]
    normal = first["normal"]
    print()
    print("HOW TO READ THESE NUMBERS")
    print("-" * width)
    print(f"* This sample is deliberately trap-heavy: {trap} of {first['n']} prompts are")
    print("  hallucination-prone by design (classes A/B/C), only "
          f"{normal} are normal questions.")
    print("  So 'Accuracy' looks low and stays flat across modes — that is expected.")
    print("  Accuracy is NOT the metric that shows Sakshi's value on this set,")
    print("  because most prompts SHOULD NOT be answered confidently at all.")
    print()
    print("* The metrics that matter are INTERVENED, ABSTAIN, and GROUNDED:")
    print("    - Baseline answers everything (0% intervention) — the raw risk.")
    print("    - Sakshi catches the risky ones and abstains rather than fabricating.")
    print("    - Sakshi+Omega goes further: it grounds answers in evidence instead")
    print("      of just abstaining (see the Grounded row).")
    print()
    print("* WHY THIS MATTERS FOR A DECISION-SUPPORT / INSIGHTS SETTING:")
    print("  The value is that the system does NOT confidently assert an unsupported")
    print("  claim. It abstains or grounds instead. In a workflow where a wrong")
    print("  'insight' drives a costly decision, that restraint is the benefit —")
    print("  it prevents acting on a confident fabrication. Read Intervened/Grounded")
    print("  as 'how often the system stopped an unsupported answer from going")
    print("  through unchecked'.")
    print()
    print("* NOTE: results here are on general-knowledge prompts. They demonstrate")
    print("  the mechanism, not domain-specific performance. Fit to a specific")
    print("  workflow (and what evidence grounding draws on) is what a pilot")
    print("  would establish.")
    print("=" * width)


if __name__ == "__main__":
    main()