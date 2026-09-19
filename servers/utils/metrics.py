"""
ML Metrics & Evaluation MCP Server
BLEU, ROUGE, accuracy, latency helpers, and simple classification metrics.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from typing import Any, Dict, List, Optional

from servers.common import create_server, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "utils-metrics",
    "Evaluation metrics for ML pipelines: BLEU, ROUGE, accuracy, F1, "
    "and simple latency / throughput measurement helpers.",
)


@mcp.tool()
def compute_accuracy(predictions: List[str], references: List[str]) -> str:
    """Exact-match accuracy between prediction and reference lists."""
    if len(predictions) != len(references):
        return safe_json({"error": "Length mismatch"})
    correct = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    acc = correct / len(predictions) if predictions else 0.0
    return safe_json({"accuracy": acc, "correct": correct, "total": len(predictions)})


@mcp.tool()
def compute_bleu(predictions: List[str], references: List[str]) -> str:
    """Corpus BLEU score (requires sacrebleu)."""
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        return safe_json({"bleu": bleu.score, "precisions": bleu.precisions, "bp": bleu.bp})
    except ImportError:
        return safe_json({"error": "sacrebleu not installed. pip install sacrebleu"})


@mcp.tool()
def compute_rouge(predictions: List[str], references: List[str]) -> str:
    """ROUGE-1/2/L scores (requires rouge_score)."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            s = scorer.score(ref, pred)
            for k in scores:
                scores[k].append(s[k].fmeasure)
        avg = {k: sum(v) / len(v) if v else 0.0 for k, v in scores.items()}
        return safe_json(avg)
    except ImportError:
        return safe_json({"error": "rouge_score not installed. pip install rouge_score"})


@mcp.tool()
def classification_report_simple(
    y_true: List[str],
    y_pred: List[str],
) -> str:
    """Simple per-class precision / recall / F1 from string labels."""
    if len(y_true) != len(y_pred):
        return safe_json({"error": "Length mismatch"})
    from collections import defaultdict
    labels = sorted(set(y_true) | set(y_pred))
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    report = {}
    for lab in labels:
        prec = tp[lab] / (tp[lab] + fp[lab]) if (tp[lab] + fp[lab]) else 0.0
        rec = tp[lab] / (tp[lab] + fn[lab]) if (tp[lab] + fn[lab]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        report[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp[lab] + fn[lab]}
    return safe_json(report)


@mcp.tool()
def compute_f1(
    predictions: List[str],
    references: List[str],
    average: str = "macro",
) -> str:
    """F1 score with configurable averaging: binary, micro, macro, weighted."""
    if len(predictions) != len(references):
        return safe_json({"error": "Length mismatch"})
    try:
        from sklearn.metrics import f1_score

        score = float(f1_score(references, predictions, average=average))
        return safe_json({"f1": score, "average": average, "total": len(predictions)})
    except Exception as e:
        return safe_json({"error": str(e), "average": average})


@mcp.tool()
def compute_perplexity(
    token_log_probs: Optional[List[float]] = None,
    token_probs: Optional[List[float]] = None,
) -> str:
    """Compute perplexity from token log-probabilities or probabilities."""
    if token_log_probs:
        avg_neg_log_prob = -sum(token_log_probs) / len(token_log_probs)
    elif token_probs:
        if any(p <= 0 or p > 1 for p in token_probs):
            return safe_json({"error": "token_probs must be between 0 and 1"})
        avg_neg_log_prob = -sum(math.log(p) for p in token_probs) / len(token_probs)
    else:
        return safe_json({"error": "Provide token_log_probs or token_probs"})
    return safe_json({
        "perplexity": float(math.exp(avg_neg_log_prob)),
        "num_tokens": len(token_log_probs or token_probs or []),
    })


@mcp.tool()
def measure_latency(func_name: str = "dummy", n_runs: int = 10) -> str:
    """Placeholder latency helper — returns timing of a no-op loop for calibration."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = sum(i * i for i in range(1000))  # tiny workload
        times.append(time.perf_counter() - t0)
    return safe_json({
        "func": func_name,
        "n_runs": n_runs,
        "mean_s": sum(times) / len(times),
        "min_s": min(times),
        "max_s": max(times),
        "note": "Replace with real model call timing in your agent loop",
    })


@mcp.tool()
def summarize_latency(
    latencies_ms: List[float],
    target_p95_ms: Optional[float] = None,
    error_count: int = 0,
) -> str:
    """Summarize observed latency samples for audit / pressure-check workflows."""
    if not latencies_ms:
        return safe_json({"error": "latencies_ms must not be empty"})
    ordered = sorted(float(v) for v in latencies_ms)
    idx = max(0, math.ceil(0.95 * len(ordered)) - 1)
    summary = {
        "count": len(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[idx],
        "error_count": error_count,
        "error_rate": error_count / len(ordered),
    }
    if target_p95_ms is not None:
        summary["target_p95_ms"] = target_p95_ms
        summary["passes_target"] = summary["p95_ms"] <= target_p95_ms
    return safe_json(summary)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
