# =============================
# AI MODEL BENCHMARK HARNESS v7
# =============================

import requests, time, threading, random, os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.colab import files

CHECKPOINT_DIR = '/content/drive/MyDrive/stop_b_benchmark'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'benchmark_results.csv')
print(f"Checkpoints will be written to: {CHECKPOINT_PATH}")

# ---- Keys ----
GOOGLE_KEY = input("Google AI Studio key: ").strip()
GROQ_KEY = input("Groq key: ").strip()
OPENROUTER_KEY = input("OpenRouter key: ").strip()

# ---- Upload and merge the corpus files ----
uploaded = files.upload()

validated_path = next(f for f in uploaded if 'validated' in f.lower())
control_path = next(f for f in uploaded if 'control' in f.lower())
selfwritten_path = next((f for f in uploaded if 'self_written' in f.lower()), None)

validated = pd.read_csv(validated_path)
control = pd.read_csv(control_path)
validated['is_khaleeji'] = True
control['is_khaleeji'] = False
frames = [validated, control]

if selfwritten_path:
    sw = pd.read_csv(selfwritten_path).rename(columns={'text': 'review_text'})
    sw['review_id'] = ['self_%03d' % i for i in range(len(sw))]
    sw['is_khaleeji'] = True
    frames.append(sw)

corpus = pd.concat(frames, ignore_index=True)
corpus = corpus.dropna(subset=['review_text', 'sentiment'])
corpus = corpus[corpus['review_text'].str.strip() != '']
corpus = corpus.drop_duplicates(subset='review_id')
print(f"Corpus: {len(corpus)} items ({corpus['is_khaleeji'].sum()} Khaleeji, "
      f"{(~corpus['is_khaleeji']).sum()} control)")

# ---- Discover real model IDs per provider, never guess slugs ----
def find_best(candidates, available):
    for c in candidates:
        matches = [m for m in available if c.lower() in m.lower()]
        if matches:
            return sorted(matches, key=len)[0], matches
    return None, []

g_resp = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={GOOGLE_KEY}")
g_resp.raise_for_status()
google_models = [m['name'].replace('models/', '') for m in g_resp.json()['models']]
gemini_id, gemini_matches = find_best(["gemini-3.5-flash", "gemini-3-flash"], google_models)

q_resp = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {GROQ_KEY}"})
q_resp.raise_for_status()
groq_models = [m['id'] for m in q_resp.json()['data']]

saba_id, saba_matches = find_best(
    ["qwen3.6-27b", "qwen3-32b", "qwen/qwen3.6", "gpt-oss-120b"], groq_models)
deepseek_id, deepseek_matches = find_best(
    ["gpt-oss-20b", "gpt-oss-120b", "llama-3.3-70b-versatile", "qwen3.6"], groq_models)

or_resp = requests.get("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {OPENROUTER_KEY}"})
or_resp.raise_for_status()
or_models = [m['id'] for m in or_resp.json()['data']]
gpt_id, gpt_matches = find_best(["openai/gpt-5.5"], or_models)
claude_id, claude_matches = find_best(["anthropic/claude-sonnet-5", "anthropic/claude-sonnet-4.5"], or_models)

RESOLVED = {
    "Gemini Flash": {"id": gemini_id, "provider": "google", "matches": gemini_matches},
    f"Groq-1 ({saba_id})": {"id": saba_id, "provider": "groq", "matches": saba_matches},
    f"Groq-2 ({deepseek_id})": {"id": deepseek_id, "provider": "groq", "matches": deepseek_matches},
    "GPT-5.5": {"id": gpt_id, "provider": "openrouter", "matches": gpt_matches},
    "Claude Sonnet": {"id": claude_id, "provider": "openrouter", "matches": claude_matches},
}

for label, info in RESOLVED.items():
    print(f"  {label} [{info['provider']}]: {info['id'] if info['id'] else 'NOT FOUND'}  "
          f"(candidates seen: {info['matches'][:4]})")

missing = [k for k, v in RESOLVED.items() if v["id"] is None]
if missing:
    print(f"Could not auto-resolve: {missing}.")
    raise SystemExit("Fix and re-run.")

n_calls = len(RESOLVED) * len(corpus)
proceed = input(f"{len(RESOLVED)} models x {len(corpus)} items = {n_calls} calls. Proceed? (y/n): ")
if proceed.strip().lower() != 'y':
    raise SystemExit("Stopped before spending anything.")

done = pd.DataFrame(columns=['review_id', 'model_label', 'predicted_sentiment', 'raw_response'])
done_keys = set()

# ---- Per-provider rate limiter, conservative margin under published free-tier caps ----
# Gemini set to 5/min (not the documented 8-15) since bursts of parallel
# requests can trigger 429s well before the nominal per-minute cap is hit.
RATE_LIMITS = {"google": 5, "groq": 25, "openrouter": 60}
_rate_lock = {p: threading.Lock() for p in RATE_LIMITS}
_last_call = {p: 0.0 for p in RATE_LIMITS}

def throttle(provider):
    min_interval = 60.0 / RATE_LIMITS[provider]
    with _rate_lock[provider]:
        wait = _last_call[provider] + min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call[provider] = time.time()

# ---- Prompt and per-provider call function ----
PROMPT_TEMPLATE = (
    "Classify the sentiment of this Arabic customer review as exactly one word: "
    "positive, negative, neutral, or mixed. Respond with only that single word, "
    "nothing else.\n\nReview: {text}"
)

def classify(review_id, text, model_label, model_id, provider):
    prompt = PROMPT_TEMPLATE.format(text=text)
    throttle(provider)
    raw = None
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            if provider == "google":
                # maxOutputTokens=200 + thinkingLevel=low prevents Gemini's
                # internal reasoning tokens from silently eating the whole
                # budget before it emits the actual answer
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GOOGLE_KEY}",
                    json={"contents": [{"parts": [{"text": prompt}]}],
                          "generationConfig": {
                              "temperature": 0,
                              "maxOutputTokens": 200,
                              "thinkingConfig": {"thinkingLevel": "low"},
                          }},
                    timeout=30,
                )
                r.raise_for_status()
                cand = r.json()['candidates'][0]
                # .get() avoids a KeyError crash if the response has no parts
                # (e.g. finishReason=SAFETY or MAX_TOKENS with empty content)
                parts = cand.get('content', {}).get('parts', [])
                raw = parts[0]['text'].strip() if parts else f"(no content, finishReason={cand.get('finishReason')})"

            elif provider == "groq":
                payload = {"model": model_id,
                           "messages": [{"role": "user", "content": prompt}],
                           "temperature": 0}
                if "gpt-oss" in model_id:
                    # gpt-oss models need explicit reasoning controls or they
                    # burn the token budget on hidden chain-of-thought
                    payload["max_completion_tokens"] = 300
                    payload["reasoning_effort"] = "low"
                    payload["reasoning_format"] = "hidden"
                elif "qwen" in model_id:
                    # qwen3.6-27b is the only Groq model that accepts
                    # reasoning_effort; "none" disables its thinking mode
                    payload["max_completion_tokens"] = 30
                    payload["reasoning_effort"] = "none"
                else:
                    payload["max_tokens"] = 10
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_KEY}"},
                    json=payload, timeout=30,
                )
                r.raise_for_status()
                raw = r.json()['choices'][0]['message']['content'].strip()
                if not raw:
                    raw = "(empty content, likely still-reasoning cutoff)"

            else:  # openrouter
                if "gpt-5.5" in model_id:
                    # GPT-5.5 rejects the legacy max_tokens/temperature params;
                    # needs max_completion_tokens + reasoning_effort instead
                    payload = {"model": model_id,
                               "messages": [{"role": "user", "content": prompt}],
                               "max_completion_tokens": 20,
                               "reasoning_effort": "none"}
                else:
                    payload = {"model": model_id,
                               "messages": [{"role": "user", "content": prompt}],
                               "max_tokens": 10, "temperature": 0}
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                    json=payload, timeout=30,
                )
                r.raise_for_status()
                raw = r.json()['choices'][0]['message']['content'].strip()
            break
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            raw = f"ERROR: {e}"
            if status == 429:
                # exponential backoff, capped at 45s, for rate-limit errors specifically
                time.sleep(min(3 * (2 ** attempt), 45))
            else:
                time.sleep(2)
        except Exception as e:
            raw = f"ERROR: {e}"
            time.sleep(2)

    low = raw.lower()
    predicted = next((s for s in ['positive', 'negative', 'neutral', 'mixed'] if s in low), 'unparseable')
    return {'review_id': review_id, 'model_label': model_label,
            'predicted_sentiment': predicted, 'raw_response': raw}

# ---- Run all calls, checkpoint every 50 ----
tasks = []
for label, info in RESOLVED.items():
    for _, row in corpus.iterrows():
        tasks.append((row['review_id'], row['review_text'], label, info['id'], info['provider']))

# Shuffle so all 5 models get worked concurrently by the 8 worker threads,
# instead of queuing behind Gemini's slow throttle first
random.shuffle(tasks)

print(f"{len(tasks)} calls to make.")

new_rows = []
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(classify, *t): t for t in tasks}
    completed = 0
    for fut in as_completed(futures):
        new_rows.append(fut.result())
        completed += 1
        if completed % 50 == 0:
            print(f"  {completed}/{len(tasks)} done")
            pd.DataFrame(new_rows).to_csv(CHECKPOINT_PATH, index=False)

all_results = pd.DataFrame(new_rows)
all_results.to_csv(CHECKPOINT_PATH, index=False)
all_results.to_csv('benchmark_results.csv', index=False)
print(f"Saved {len(all_results)} total results to Drive ({CHECKPOINT_PATH}) and local disk.")
files.download('benchmark_results.csv')

# ---- Score accuracy: overall, and Khaleeji vs control ----
scored = all_results.merge(corpus[['review_id', 'sentiment', 'is_khaleeji']], on='review_id', how='left')
scored['correct'] = scored['predicted_sentiment'] == scored['sentiment']

summary = scored.groupby(['model_label', 'is_khaleeji'])['correct'].agg(['mean', 'count']).reset_index()
summary.columns = ['model', 'is_khaleeji', 'accuracy', 'n']
summary['accuracy'] = (summary['accuracy'] * 100).round(1)
summary = summary.sort_values(['model', 'is_khaleeji'], ascending=[True, False])

print("Accuracy by model, Khaleeji vs control:")
print(summary.to_string(index=False))

summary.to_csv('benchmark_summary.csv', index=False)
files.download('benchmark_summary.csv')

# ---- Flag rows still unparseable/erroring after retries ----
bad = all_results[all_results['predicted_sentiment'] == 'unparseable']
if len(bad):
    print(f"{len(bad)} rows still unparseable after 5 retries each. By model:")
    print(bad['model_label'].value_counts().to_string())
    bad.to_csv('still_unparseable.csv', index=False)
    files.download('still_unparseable.csv')