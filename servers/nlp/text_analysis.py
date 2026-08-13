"""
NLP Text Analysis MCP Server
Tools: classify_text, extract_entities, sentiment_analysis, summarize, extract_keywords
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from servers.common import create_server, get_device, safe_json

logger = logging.getLogger(__name__)
mcp = create_server(
    "nlp-text-analysis",
    "NLP tools for classification, NER, sentiment, summarization and keyword extraction. "
    "Models are loaded on first use and cached.",
)

_pipeline_cache: Dict[str, Any] = {}


def _get_pipeline(task: str, model: Optional[str] = None):
    key = f"{task}:{model or 'default'}"
    if key not in _pipeline_cache:
        from transformers import pipeline
        device = 0 if get_device() == "cuda" else -1
        defaults = {
            "sentiment-analysis": "distilbert-base-uncased-finetuned-sst-2-english",
            "ner": "dslim/bert-base-NER",
            "summarization": "facebook/bart-large-cnn",
            "text-classification": "facebook/bart-large-mnli",
        }
        model_id = model or defaults.get(task, "distilbert-base-uncased")
        _pipeline_cache[key] = pipeline(task, model=model_id, device=device)
        logger.info("Loaded pipeline %s -> %s", task, model_id)
    return _pipeline_cache[key]


@mcp.tool()
def classify_text(
    text: str,
    labels: Optional[List[str]] = None,
    model: Optional[str] = None,
) -> str:
    """Zero-shot or standard text classification.
    If labels are provided, uses zero-shot classification against those candidate labels.
    """
    if labels:
        pipe = _get_pipeline("zero-shot-classification", model)
        result = pipe(text, candidate_labels=labels)
    else:
        pipe = _get_pipeline("text-classification", model)
        result = pipe(text)
    return safe_json(result)


@mcp.tool()
def extract_entities(
    text: str,
    model: Optional[str] = None,
    aggregation_strategy: str = "simple",
) -> str:
    """Named Entity Recognition (NER). Returns entities with labels, scores and character spans."""
    pipe = _get_pipeline("ner", model)
    try:
        result = pipe(text, aggregation_strategy=aggregation_strategy)
    except TypeError:
        result = pipe(text)
    return safe_json(result)


@mcp.tool()
def sentiment_analysis(text: str, model: Optional[str] = None) -> str:
    """Sentiment analysis (positive / negative / neutral scores)."""
    pipe = _get_pipeline("sentiment-analysis", model)
    result = pipe(text)
    return safe_json(result)


@mcp.tool()
def summarize(
    text: str,
    max_length: int = 130,
    min_length: int = 30,
    model: Optional[str] = None,
) -> str:
    """Abstractive summarization of the input text."""
    pipe = _get_pipeline("summarization", model)
    result = pipe(text, max_length=max_length, min_length=min_length, do_sample=False)
    return safe_json(result)


@mcp.tool()
def extract_keywords(text: str, top_k: int = 10) -> str:
    """Simple keyword / keyphrase extraction using YAKE-style scoring fallback or KeyBERT if available."""
    try:
        from keybert import KeyBERT
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(text, top_n=top_k)
        return safe_json([{"keyword": k, "score": float(s)} for k, s in keywords])
    except ImportError:
        import re
        from collections import Counter
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        stop = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our", "out", "has", "his", "how", "its", "may", "new", "now", "old", "see", "two", "way", "who", "boy", "did", "get", "let", "put", "say", "she", "too", "use"}
        filtered = [w for w in words if w not in stop]
        counts = Counter(filtered).most_common(top_k)
        return safe_json([{"keyword": k, "score": c} for k, c in counts])


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
