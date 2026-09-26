import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from config import DataConfig, GlobalConfig
from data.filters.harmful_filter import classify_nsfw, classify_toxic_speech
from data.filters.language_identification import identify_language
from data.filters.quality_classifier import classify_quality

SNIPPET_LENGTH = 300

class DocScore(NamedTuple):
    doc_id: str
    text_snippet: str
    quality_label: str
    quality_score: float
    nsfw_label: str
    nsfw_score: float
    toxic_label: str
    toxic_score: float
    lang_label: str
    lang_score: float

def score_document(doc_id: str, text: str) -> DocScore:
    """
    Run a document through every diagnostic classifier and collect raw scores.

    This never filters or drops a document; it only measures. The active
    filtering pipeline lives in `data/preprocessing_pipeline.py`.

    Args:
        doc_id (str): Identifier of the document, used to trace samples back.
        text (str): Document text to score.

    Returns:
        DocScore: Label/score pairs from every classifier, plus a text snippet.
    """
    quality_label, quality_score = classify_quality(text=text)
    nsfw_label, nsfw_score = classify_nsfw(text=text)
    toxic_label, toxic_score = classify_toxic_speech(text=text)
    lang_label, lang_score = identify_language(text=text)

    return DocScore(
        doc_id=doc_id,
        text_snippet=text[:SNIPPET_LENGTH],
        quality_label=quality_label,
        quality_score=quality_score,
        nsfw_label=nsfw_label,
        nsfw_score=nsfw_score,
        toxic_label=toxic_label,
        toxic_score=toxic_score,
        lang_label=lang_label,
        lang_score=lang_score,
    )

def load_documents(input_path: Path, limit: int | None) -> list[tuple[str, str]]:
    """
    Load (doc_id, text) pairs from a JSONL file.

    Args:
        input_path (Path): Path to a JSONL file with "id" and "text" fields per line.
        limit (int | None): Maximum number of documents to load, or None for all.

    Returns:
        list[tuple[str, str]]: Loaded (doc_id, text) pairs, in file order.
    """
    documents = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            if limit is not None and line_number >= limit:
                break
            record = json.loads(line)
            documents.append((str(record.get("id", line_number)), record["text"]))
    return documents

def _percentile(sorted_values: list[float], pct: float) -> float:
    idx = min(len(sorted_values) - 1, max(0, round(pct / 100 * (len(sorted_values) - 1))))
    return sorted_values[idx]

def _score_distribution(scores: list[float]) -> dict:
    if not scores:
        return {"count": 0}

    ordered = sorted(scores)
    return {
        "count": len(scores),
        "mean": statistics.fmean(scores),
        "std": statistics.pstdev(scores),
        "min": ordered[0],
        "max": ordered[-1],
        "p10": _percentile(ordered, 10),
        "p25": _percentile(ordered, 25),
        "p50": _percentile(ordered, 50),
        "p75": _percentile(ordered, 75),
        "p90": _percentile(ordered, 90),
        "p99": _percentile(ordered, 99),
    }

def _doc_score_to_sample(doc_score: DocScore, label_attr: str, score_attr: str) -> dict:
    return {
        "doc_id": doc_score.doc_id,
        "label": getattr(doc_score, label_attr),
        "score": getattr(doc_score, score_attr),
        "text_snippet": doc_score.text_snippet,
    }

def summarize_quality(doc_scores: list[DocScore], sample_size: int) -> dict:
    """
    Aggregate the quality classifier's scores over a batch of documents.

    Args:
        doc_scores (list[DocScore]): Scored documents.
        sample_size (int): Number of extreme examples to keep per side.

    Returns:
        dict: Label counts/fractions, score distribution, the fraction that
        would have been dropped by this classifier at its configured
        threshold, and qualitative samples from both extremes.
    """
    total = len(doc_scores)
    label_counts = Counter(d.quality_label for d in doc_scores)
    scores = [d.quality_score for d in doc_scores]

    would_flag = sum(
        1 for d in doc_scores
        if d.quality_label == "low_quality" and d.quality_score >= DataConfig.quality_filter_threshold
    )

    low_quality_sample = sorted(
        (d for d in doc_scores if d.quality_label == "low_quality"),
        key=lambda d: d.quality_score, reverse=True,
    )[:sample_size]
    high_quality_sample = sorted(
        (d for d in doc_scores if d.quality_label != "low_quality"),
        key=lambda d: d.quality_score, reverse=True,
    )[:sample_size]

    return {
        "label_counts": dict(label_counts),
        "label_fractions": {k: v / total for k, v in label_counts.items()} if total else {},
        "score_distribution": _score_distribution(scores),
        "flag_rate": would_flag / total if total else 0.0,
        "threshold": DataConfig.quality_filter_threshold,
        "samples": {
            "most_confident_low_quality": [
                _doc_score_to_sample(d, "quality_label", "quality_score") for d in low_quality_sample
            ],
            "most_confident_high_quality": [
                _doc_score_to_sample(d, "quality_label", "quality_score") for d in high_quality_sample
            ],
        },
    }

def summarize_nsfw(doc_scores: list[DocScore], sample_size: int) -> dict:
    """
    Aggregate the NSFW classifier's scores over a batch of documents.

    Args:
        doc_scores (list[DocScore]): Scored documents.
        sample_size (int): Number of extreme examples to keep per side.

    Returns:
        dict: Label counts/fractions, score distribution, the fraction that
        would have been dropped by this classifier at its configured
        threshold, and qualitative samples from both extremes.
    """
    total = len(doc_scores)
    label_counts = Counter(d.nsfw_label for d in doc_scores)
    scores = [d.nsfw_score for d in doc_scores]

    would_flag = sum(
        1 for d in doc_scores
        if d.nsfw_label == "nsfw" and d.nsfw_score >= DataConfig.nsfw_filter_threshold
    )

    nsfw_sample = sorted(
        (d for d in doc_scores if d.nsfw_label == "nsfw"),
        key=lambda d: d.nsfw_score, reverse=True,
    )[:sample_size]
    sfw_sample = sorted(
        (d for d in doc_scores if d.nsfw_label != "nsfw"),
        key=lambda d: d.nsfw_score, reverse=True,
    )[:sample_size]

    return {
        "label_counts": dict(label_counts),
        "label_fractions": {k: v / total for k, v in label_counts.items()} if total else {},
        "score_distribution": _score_distribution(scores),
        "flag_rate": would_flag / total if total else 0.0,
        "threshold": DataConfig.nsfw_filter_threshold,
        "samples": {
            "most_confident_nsfw": [
                _doc_score_to_sample(d, "nsfw_label", "nsfw_score") for d in nsfw_sample
            ],
            "most_confident_sfw": [
                _doc_score_to_sample(d, "nsfw_label", "nsfw_score") for d in sfw_sample
            ],
        },
    }

def summarize_toxic(doc_scores: list[DocScore], sample_size: int) -> dict:
    """
    Aggregate the hate-speech classifier's scores over a batch of documents.

    Args:
        doc_scores (list[DocScore]): Scored documents.
        sample_size (int): Number of extreme examples to keep per side.

    Returns:
        dict: Label counts/fractions, score distribution, the fraction that
        would have been dropped by this classifier at its configured
        threshold, and qualitative samples from both extremes.
    """
    total = len(doc_scores)
    label_counts = Counter(d.toxic_label for d in doc_scores)
    scores = [d.toxic_score for d in doc_scores]

    would_flag = sum(
        1 for d in doc_scores
        if d.toxic_label == "toxic" and d.toxic_score >= DataConfig.hatespeech_filter_threshold
    )

    toxic_sample = sorted(
        (d for d in doc_scores if d.toxic_label == "toxic"),
        key=lambda d: d.toxic_score, reverse=True,
    )[:sample_size]
    non_toxic_sample = sorted(
        (d for d in doc_scores if d.toxic_label != "toxic"),
        key=lambda d: d.toxic_score, reverse=True,
    )[:sample_size]

    return {
        "label_counts": dict(label_counts),
        "label_fractions": {k: v / total for k, v in label_counts.items()} if total else {},
        "score_distribution": _score_distribution(scores),
        "flag_rate": would_flag / total if total else 0.0,
        "threshold": DataConfig.hatespeech_filter_threshold,
        "samples": {
            "most_confident_toxic": [
                _doc_score_to_sample(d, "toxic_label", "toxic_score") for d in toxic_sample
            ],
            "most_confident_non_toxic": [
                _doc_score_to_sample(d, "toxic_label", "toxic_score") for d in non_toxic_sample
            ],
        },
    }

def summarize_language(doc_scores: list[DocScore], sample_size: int) -> dict:
    """
    Aggregate the language-ID classifier's scores over a batch of documents.

    Args:
        doc_scores (list[DocScore]): Scored documents.
        sample_size (int): Number of extreme examples to keep per side.

    Returns:
        dict: Label counts/fractions, score distribution, the fraction that
        would have been dropped by the active filter's language check, and
        qualitative samples from both extremes.
    """
    total = len(doc_scores)
    label_counts = Counter(d.lang_label for d in doc_scores)
    scores = [d.lang_score for d in doc_scores]

    would_flag = sum(
        1 for d in doc_scores
        if d.lang_label != "en" or d.lang_score < DataConfig.language_confidence_filter_threshold
    )

    non_english_sample = sorted(
        (d for d in doc_scores if d.lang_label != "en"),
        key=lambda d: d.lang_score, reverse=True,
    )[:sample_size]
    english_sample = sorted(
        (d for d in doc_scores if d.lang_label == "en"),
        key=lambda d: d.lang_score, reverse=True,
    )[:sample_size]

    return {
        "label_counts": dict(label_counts),
        "label_fractions": {k: v / total for k, v in label_counts.items()} if total else {},
        "score_distribution": _score_distribution(scores),
        "flag_rate": would_flag / total if total else 0.0,
        "threshold": DataConfig.language_confidence_filter_threshold,
        "samples": {
            "most_confident_non_english": [
                _doc_score_to_sample(d, "lang_label", "lang_score") for d in non_english_sample
            ],
            "most_confident_english": [
                _doc_score_to_sample(d, "lang_label", "lang_score") for d in english_sample
            ],
        },
    }

def measure_dataset(input_path: Path, stage: str, seed: int, sample_size: int, limit: int | None) -> dict:
    """
    Score a batch of documents and summarize every diagnostic classifier.

    Args:
        input_path (Path): JSONL file with "id" and "text" fields per line.
        stage (str): Label for where in the pipeline this sample was taken
            from, e.g. "pre_filter" or "post_filter_dedup".
        seed (int): Seed used to draw the input sample upstream, recorded
            here for reproducibility.
        sample_size (int): Number of extreme examples to keep per classifier side.
        limit (int | None): Maximum number of documents to score, or None for all.

    Returns:
        dict: Run metadata plus a per-classifier summary, ready to json.dump.
    """
    documents = load_documents(input_path, limit=limit)
    doc_scores = [score_document(doc_id=doc_id, text=text) for doc_id, text in documents]

    return {
        "stage": stage,
        "source_file": str(input_path),
        "sample_size": len(doc_scores),
        "seed": seed,
        "classifiers": {
            "quality": summarize_quality(doc_scores, sample_size),
            "nsfw": summarize_nsfw(doc_scores, sample_size),
            "toxic_speech": summarize_toxic(doc_scores, sample_size),
            "language": summarize_language(doc_scores, sample_size),
        },
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a document sample with the diagnostic classifiers "
        "(quality, NSFW, hate speech, language ID) without filtering anything."
    )
    parser.add_argument("--input", type=Path, required=True, help="JSONL file with 'id' and 'text' fields.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the JSON report.")
    parser.add_argument("--limit", type=int, default=None, help="Max documents to score.")
    parser.add_argument(
        "--stage", type=str, required=True,
        help="Where this sample was taken from, e.g. 'pre_filter' or 'post_filter_dedup'.",
    )
    parser.add_argument("--sample-size", type=int, default=10, help="Extreme examples to keep per classifier side.")
    parser.add_argument("--seed", type=int, default=GlobalConfig.seed)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    report = measure_dataset(
        input_path=args.input,
        stage=args.stage,
        seed=args.seed,
        sample_size=args.sample_size,
        limit=args.limit,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
