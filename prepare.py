"""Build the public data files from the private corpus files

Usage: python prepare.py (needs the private corpus files, which are not in this repo)
Reads the three corpus files in private/ and writes data/labels.csv (no review text)
and data/written_reviews.csv (the native-written reviews with text)
"""

from pathlib import Path

import pandas as pd

PRIVATE = Path("private")
DATA = Path("data")


def load_corpus() -> pd.DataFrame:
    """Merge the three corpus files into one table with one row per review"""
    khaleeji = pd.read_csv(PRIVATE / "validated_khaleeji_worklist.csv")
    khaleeji["group"] = "khaleeji"
    khaleeji["source"] = "mined"
    other = pd.read_csv(PRIVATE / "control_non_khaleeji.csv")
    other["group"] = "other"
    other["source"] = "mined"
    written = pd.read_csv(PRIVATE / "self_written_corpus.csv").rename(columns={"text": "review_text"})
    written["review_id"] = [f"self_{i:03d}" for i in range(len(written))]
    written["group"] = "khaleeji"
    written["source"] = "written"
    written["n_raters"] = 1
    corpus = pd.concat([khaleeji, other, written], ignore_index=True)
    corpus = corpus.dropna(subset=["review_text", "sentiment"])
    corpus = corpus[corpus["review_text"].str.strip() != ""]
    return corpus.drop_duplicates(subset="review_id")


def main() -> None:
    corpus = load_corpus()
    DATA.mkdir(exist_ok=True)
    columns = ["review_id", "group", "source", "app", "sentiment", "n_raters"]
    labels = corpus[columns].assign(n_raters=corpus["n_raters"].astype(int))
    labels.to_csv(DATA / "labels.csv", index=False)
    written = corpus[corpus["source"] == "written"].copy()
    written["country"] = written["origin"].str.split(" (", regex=False).str[0]
    public = ["review_id", "participant", "country", "app", "sentiment", "review_text"]
    written[public].to_csv(DATA / "written_reviews.csv", index=False, encoding="utf-8")
    print(f"{len(labels)} labels and {len(written)} written reviews saved to {DATA}/")


if __name__ == "__main__":
    main()
