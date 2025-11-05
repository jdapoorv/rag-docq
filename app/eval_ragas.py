from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
from ragas import evaluate
from app.rag_chain import answer

# create eval set (question, ground_truth, contexts)
eval_qas = [
    {"question": "What is X?", "ground_truth": "X is ..."},
    # add 10–20 items based on your docs
]

records = []
for item in eval_qas:
    ans, ctx_docs = answer(item["question"])
    records.append({
        "question": item["question"],
        "answers": ans,
        "contexts": [d.page_content for d in ctx_docs],
        "ground_truth": item["ground_truth"]
    })

ds = Dataset.from_list(records)
res = evaluate(
    ds,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness],
)
print(res)
