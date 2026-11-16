# ====================================
# JAIS 2 8B via LM Studio local server
# ====================================

import time

import pandas as pd
import requests

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"

MODEL_ID = "jais-2-8b-chat"

PROMPT_TEMPLATE = (
    "Classify the sentiment of this Arabic customer review as exactly one word: "
    "positive, negative, neutral, or mixed. Respond with only that single word, "
    "nothing else.\n\nReview: {text}"
)

# ---- Load the same 3 corpus files from local disk ----
validated = pd.read_csv("validated_khaleeji_worklist.csv")
validated["is_khaleeji"] = True
control = pd.read_csv("control_non_khaleeji.csv")
control["is_khaleeji"] = False
sw = pd.read_csv("self_written_corpus.csv").rename(columns={"text": "review_text"})
sw["review_id"] = ["self_%03d" % i for i in range(len(sw))]
sw["is_khaleeji"] = True

corpus = pd.concat([validated, control, sw], ignore_index=True)
corpus = corpus.dropna(subset=["review_text", "sentiment"])
corpus = corpus[corpus["review_text"].str.strip() != ""]
corpus = corpus.drop_duplicates(subset="review_id")
print(f"Corpus: {len(corpus)} items")

# ---- Sanity check: confirm LM Studio is up and the model id is right ----
try:
    check = requests.get("http://localhost:1234/v1/models", timeout=5)
    check.raise_for_status()
    live_ids = [m["id"] for m in check.json().get("data", [])]
    print(f"LM Studio reports these loaded model(s): {live_ids}")
    if MODEL_ID not in live_ids:
        raise SystemExit(
            f"MODEL_ID '{MODEL_ID}' not in {live_ids}. "
            f"Fix the MODEL_ID variable above and re-run."
        )
except requests.exceptions.ConnectionError:
    raise SystemExit(
        "Can't reach http://localhost:1234. Start the Local Server in "
        "LM Studio's Developer tab first, then re-run."
    )

# ---- Classify sequentially, local inference has no rate limit to respect ----
rows = []
for i, row in corpus.iterrows():
    prompt = PROMPT_TEMPLATE.format(text=row["review_text"])
    try:
        r = requests.post(
            LMSTUDIO_URL,
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0,
            },
            timeout=60,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raw = f"ERROR: {e}"

    low = raw.lower()
    predicted = next(
        (s for s in ["positive", "negative", "neutral", "mixed"] if s in low),
        "unparseable",
    )
    rows.append(
        {
            "review_id": row["review_id"],
            "model_label": "Jais 2 8B",
            "predicted_sentiment": predicted,
            "raw_response": raw,
        }
    )

    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(corpus)} done")
        pd.DataFrame(rows).to_csv("jais_results.csv", index=False)

results = pd.DataFrame(rows)
results.to_csv("jais_results.csv", index=False)
print(f"\nSaved {len(results)} results to jais_results.csv")

# ---- Score ----
scored = results.merge(
    corpus[["review_id", "sentiment", "is_khaleeji"]], on="review_id", how="left"
)
scored["correct"] = scored["predicted_sentiment"] == scored["sentiment"]
summary = scored.groupby("is_khaleeji")["correct"].agg(["mean", "count"]).reset_index()
summary.columns = ["is_khaleeji", "accuracy", "n"]
summary["accuracy"] = (summary["accuracy"] * 100).round(1)
print("\nJais 2 8B accuracy, Khaleeji vs control:")
print(summary.to_string(index=False))
summary.to_csv("jais_summary.csv", index=False)
