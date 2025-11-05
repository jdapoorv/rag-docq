# app/eval_ragas.py
import json, time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

from app.rag_chain import answer  # uses your existing pipeline
from app.config import settings

# RAGAS (ensure it's in requirements)
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy, context_recall, context_precision
)

@dataclass
class EvalItem:
    question: str
    ground_truth: str | None = None
    source: str | None = None

def load_eval_set(path: str) -> List[EvalItem]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            items.append(EvalItem(
                question=obj["question"],
                ground_truth=obj.get("ground_truth"),
                source=obj.get("source"),
            ))
    return items

def run_batch_eval(items: List[EvalItem], k_retrieved: int = 8) -> Dict[str, Any]:
    # Collect predictions and contexts via your pipeline
    results = []
    latencies = []
    for i, it in enumerate(items, start=1):
        t0 = time.time()
        ans, docs = answer(it.question)   # your function returns (answer_text, [docs])
        lat = time.time() - t0
        latencies.append(lat)
        contexts = [d.page_content for d in docs]
        results.append({
            "question": it.question,
            "answer": ans,
            "contexts": contexts,
            "ground_truth": it.ground_truth or "",  # ragas expects strings
        })

    # Build a ragas dataset (list of dicts)
    ragas_input = [{
        "question": r["question"],
        "answer": r["answer"],
        "contexts": r["contexts"],
        "ground_truth": r["ground_truth"],
    } for r in results]

    # Evaluate with RAGAS. Uses your default OpenAI model as the judge.
    # You can control model via env: OPENAI_MODEL or set an explicit judge here.
    ragas_report = evaluate(
        ragas_input,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    # Simple retrieval metrics: Recall@k = did any context contain GT string?
    # (Only computed if ground_truth provided)
    hit, total = 0, 0
    for r in results:
        gt = (r["ground_truth"] or "").strip()
        if not gt:
            continue
        total += 1
        blob = " ".join(r["contexts"]).lower()
        if gt[:80].lower() in blob:  # naive substring check; OK for demos
            hit += 1
    recall_at_k = (hit / total) if total > 0 else None

    # Latency stats
    import numpy as np
    lat_p50 = float(np.median(latencies)) if latencies else 0.0
    lat_p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

    return {
        "ragas": ragas_report,  # this has per-metric means and distributions
        "recall_at_k": recall_at_k,
        "lat_p50": lat_p50,
        "lat_p95": lat_p95,
        "n": len(items),
        "raw": results,
    }

def save_summary(report: Dict[str, Any], outdir="data/eval"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    # ragas_report has a .to_pandas() method if needed; we can dump json for now
    with open(Path(outdir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "n": report["n"],
            "recall_at_k": report["recall_at_k"],
            "lat_p50": report["lat_p50"],
            "lat_p95": report["lat_p95"],
            "ragas_means": {m.name: m.score for m in report["ragas"].metrics},
        }, f, indent=2)
    with open(Path(outdir) / "details.jsonl", "w", encoding="utf-8") as f:
        for row in report["raw"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/eval.jsonl")
    args = ap.parse_args()

    items = load_eval_set(args.file)
    rep = run_batch_eval(items)
    save_summary(rep)
    print("Done. Saved to data/eval/")
