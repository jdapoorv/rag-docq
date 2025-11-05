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
    # 1) Run your real pipeline to collect answers + contexts
    results = []
    latencies = []
    for it in items:
        t0 = time.time()
        ans, docs = answer(it.question)
        latencies.append(time.time() - t0)
        contexts = [d.page_content for d in docs]
        results.append(
            {
                "question": it.question,
                "answer": ans or "",
                "contexts": contexts,               # list[str]
                "ground_truth": it.ground_truth or ""  # ragas expects a string (can be empty)
            }
        )

    # 2) Build a HF Dataset for ragas (NOT a plain list)
    import pandas as pd
    from datasets import Dataset as HFDataset

    df = pd.DataFrame(results)
    ds = HFDataset.from_pandas(df)

    # 3) Choose metrics (only use GT-dependent metrics if any GT present)
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
    has_gt = any(bool((r.get("ground_truth") or "").strip()) for r in results)
    metrics = [faithfulness, answer_relevancy] + ([context_recall, context_precision] if has_gt else [])

    # 4) Evaluate with ragas
    ragas_report = evaluate(ds, metrics=metrics)

    # 5) Naive retrieval Recall@k using GT substring (only if GT present)
    recall_at_k = None
    if has_gt:
        hit, total = 0, 0
        for r in results:
            gt = (r["ground_truth"] or "").strip()
            if not gt:
                continue
            total += 1
            blob = " ".join(r["contexts"]).lower()
            if gt[:80].lower() in blob:
                hit += 1
        recall_at_k = (hit / total) if total else None

    # 6) Latency stats
    import numpy as np
    lat_p50 = float(np.median(latencies)) if latencies else 0.0
    lat_p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

    return {
        "ragas": ragas_report,
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
