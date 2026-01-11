import sys
from pathlib import Path
import time
import pandas as pd

# -------------------------------------------------
# Path setup
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from rag_core import rag_answer

EVAL_FILE = Path(__file__).parent / "eval_questions.csv"
OUT_FILE = Path(__file__).parent / "eval_results.csv"


# -------------------------------------------------
# Evaluation logic
# -------------------------------------------------
def evaluate_answer(answer, sources, expected_behavior):
    """
    expected_behavior: 'rag', 'rag_answer', 'calculation', or 'refusal'
    """
    if expected_behavior in ["rag", "rag_answer"]:
        grounding_correct = bool(sources)
        accuracy = 1 if grounding_correct else 0

    elif expected_behavior in ["calculation", "refusal"]:
        grounding_correct = not bool(sources)
        accuracy = 1 if grounding_correct else 0

    else:
        grounding_correct = False
        accuracy = 0

    return accuracy, int(grounding_correct)


# -------------------------------------------------
# Run evaluation
# -------------------------------------------------
def run_eval():
    df = pd.read_csv(EVAL_FILE)
    results = []

    for _, row in df.iterrows():
        question = row["question"]
        expected_behavior = row["expected_behavior"]

        start = time.time()
        answer, sources = rag_answer(question)
        latency = round(time.time() - start, 2)

        accuracy, grounding_correct = evaluate_answer(
            answer, sources, expected_behavior
        )

        results.append({
            "question": question,
            "answer": answer,
            "num_sources": len(sources),
            "latency_sec": latency,
            "accuracy": accuracy,
            "grounding_correct": grounding_correct
        })

    pd.DataFrame(results).to_csv(OUT_FILE, index=False)
    print("Evaluation completed successfully")


if __name__ == "__main__":
    run_eval()
