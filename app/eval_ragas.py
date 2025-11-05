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
    # 1) Collect answers + contexts via your real pipeline
    results = []
    latencies = []
    for it in items:
        t0 = time.time()
        ans, docs = answer(it.question)
        latencies.append(time.time() - t0)
        contexts = [d.page_content for d in docs]
        results.append({
            "question": it.question,
            "answer": ans or "",
            "contexts": contexts,                         # list[str]
            "ground_truth": (it.ground_truth or ""),      # string (can be empty)
        })

    import pandas as pd
    from datasets import Dataset as HFDataset
    df_all = pd.DataFrame(results)

    # 2) Metrics (skip GT-dependent if no GT present)
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
    has_gt = any(bool((r.get("ground_truth") or "").strip()) for r in results)
    metrics = [faithfulness, answer_relevancy] + ([context_recall, context_precision] if has_gt else [])

    # 3) Judge LLM with conservative settings (avoid timeouts)
    from langchain_openai import ChatOpenAI
    from app.config import settings
    judge = ChatOpenAI(
        model=settings.OPENAI_MODEL,   # e.g., gpt-4o-mini
        temperature=0,
        timeout=45,                    # seconds
        max_retries=2,
    )

    # 4) Evaluate in tiny batches sequentially (avoid parallel timeouts)
    from ragas import evaluate
    score_frames = []
    BATCH = 4  # keep small on Cloud
    for i in range(0, len(df_all), BATCH):
        sub_df = df_all.iloc[i:i+BATCH].reset_index(drop=True)
        ds = HFDataset.from_pandas(sub_df)
        try:
            sub_res = evaluate(ds, metrics=metrics, llm=judge)
            score_frames.append(sub_res.to_pandas())   # << use pandas, not .metrics
        except Exception as e:
            # keep going; record a placeholder for visibility
            score_frames.append(pd.DataFrame([{"metric": "error", "score": 0.0, "detail": str(e)}]))

    df_scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame(columns=["metric","score"])

    # 5) Simple retrieval Recall@k via substring (only if GT present)
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
        "ragas_df": df_scores,        # pandas DataFrame of metric scores
        "recall_at_k": recall_at_k,
        "lat_p50": lat_p50,
        "lat_p95": lat_p95,
        "n": len(items),
        "raw": results,
    }

def save_summary(report: Dict[str, Any], outdir="data/eval"):
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Aggregate means per metric from pandas df
    import pandas as pd
    df_scores = report.get("ragas_df")
    if isinstance(df_scores, pd.DataFrame) and not df_scores.empty:
        means = df_scores.groupby("metric")["score"].mean().to_dict()
    else:
        means = {}

    with open(Path(outdir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "n": report["n"],
            "recall_at_k": report["recall_at_k"],
            "lat_p50": report["lat_p50"],
            "lat_p95": report["lat_p95"],
            "ragas_means": means,
        }, f, indent=2)

    # details file
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
