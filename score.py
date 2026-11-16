"""Score one or more predictions files against the labels and print the tables

Usage: python score.py results/predictions_2026-07.csv [more predictions files]
A later file overrides an earlier one for the same review and model
Writes a Markdown report next to the first file
"""

import math
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

LABELS = Path("data/labels.csv")
HARD = ["mixed", "neutral"]
NAMES = {
    "Groq-1 (qwen/qwen3.6-27b)": "Qwen 3.6 27B",
    "Groq-2 (openai/gpt-oss-20b)": "GPT OSS 20B",
    "Claude Sonnet": "Claude Sonnet 5",
    "GPT-5.5": "GPT 5.5",
    "Gemini Flash": "Gemini 3.5 Flash",
    "Jais 2 8B": "Jais 2 8B (local)",
}


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for k correct out of n, as percentages"""
    z = 1.96
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (centre - half), 100 * (centre + half)


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the two discordant counts"""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def load(paths: list[Path]) -> pd.DataFrame:
    """Join predictions to labels and mark each row correct or not"""
    labels = pd.read_csv(LABELS)
    preds = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    preds = preds.drop_duplicates(subset=["review_id", "model_label"], keep="last")
    preds["model"] = preds["model_label"].replace(NAMES)
    scored = preds.merge(labels, on="review_id", how="inner")
    scored["correct"] = scored["predicted_sentiment"] == scored["sentiment"]
    return scored


def table(scored: pd.DataFrame, mask: pd.Series, title: str) -> list[str]:
    """One Markdown table of correct, total, accuracy and interval per model for the rows in mask"""
    rows: list[tuple[float, str]] = []
    for model, part in scored[mask].groupby("model"):
        k, n = int(part["correct"].sum()), len(part)
        low, high = wilson(k, n)
        rows.append((k / n, f"| {model} | {k} of {n} | {100 * k / n:.1f}% | {low:.1f} to {high:.1f} |"))
    rows.sort(reverse=True)
    head = [f"### {title}", "", "| Model | Correct | Accuracy | 95% interval |", "|---|---|---|---|"]
    return [*head, *(row for _, row in rows), ""]


def pairs(scored: pd.DataFrame, mask: pd.Series, title: str) -> list[str]:
    """Exact McNemar test for every pair of models on the rows in mask, with a Bonferroni correction"""
    wide = scored[mask].pivot(index="review_id", columns="model", values="correct")
    tests: list[tuple[float, str, str, int, int]] = []
    for a, b in combinations(list(wide.columns), 2):
        both = wide[[a, b]].dropna().astype(bool)
        a_only = int((both[a] & ~both[b]).sum())
        b_only = int((~both[a] & both[b]).sum())
        tests.append((mcnemar(a_only, b_only), a, b, a_only, b_only))
    tests.sort()
    out = [f"### {title}", "", "| Pair | Only first right | Only second right | p | p after Bonferroni |", "|---|---|---|---|---|"]
    for p, a, b, a_only, b_only in tests[:5]:
        out.append(f"| {a} vs {b} | {a_only} | {b_only} | {p:.4f} | {min(1.0, p * len(tests)):.3f} |")
    return [*out, ""]


def main() -> None:
    paths = [Path(arg) for arg in sys.argv[1:]]
    scored = load(paths)
    hard = scored["sentiment"].isin(HARD)
    written = scored["source"] == "written"
    everything = pd.Series(True, index=scored.index)
    counts = scored.groupby("model")["review_id"].nunique().to_dict()
    failed = scored[scored["predicted_sentiment"] == "unparseable"].groupby("model").size().to_dict()
    lines = [f"# KhaleejiBench report for {paths[0].name}", "", f"Reviews scored per model: {counts}", ""]
    lines += [f"Unparseable answers per model: {failed}", ""]
    lines += table(scored, everything, "All reviews")
    lines += table(scored, hard, "Hard cases: true label mixed or neutral")
    lines += table(scored, hard & written, "Hard cases, native-written reviews only")
    lines += table(scored, hard & ~written, "Hard cases, mined reviews only")
    lines += table(scored, ~hard & (scored["group"] == "khaleeji"), "Clear positive and negative, Khaleeji")
    lines += table(scored, ~hard & (scored["group"] == "other"), "Clear positive and negative, non-Khaleeji")
    lines += pairs(scored, hard, "Closest-to-significant model pairs on hard cases")
    lines += pairs(scored, hard & written, "Same test on native-written hard cases only")
    report = "\n".join(lines)
    print(report)
    name = paths[0].stem.replace("predictions", "report") + ".md"
    paths[0].with_name(name).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
