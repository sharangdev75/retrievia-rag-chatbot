import sys
from pathlib import Path
import time
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from rag_core import rag_answer

EVAL_FILE = Path(__file__).parent / "eval_questions.csv"
OUT_FILE = Path(__file__).parent / "eval_results.csv"

def run_eval():
    df = pd.read_csv(EVAL_FILE)
    results = []

    for _, row in df.iterrows():
        q = row["question"]

        start = time.time()
        answer, sources = rag_answer(q)
        latency = round(time.time() - start, 2)

        results.append({
            "question": q,
            "answer": answer,
            "has_sources": bool(sources),
            "num_sources": len(sources),
            "latency_sec": latency
        })

    pd.DataFrame(results).to_csv(OUT_FILE, index=False)
    print("Evaluation completed successfully")

if __name__ == "__main__":
    run_eval()
