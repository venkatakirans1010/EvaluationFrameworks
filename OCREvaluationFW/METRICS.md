# OCR Evaluation Framework Metrics Documentation

This document describes every metric as implemented in this framework, including preprocessing, computation, aggregation, and interpretation. Each section includes the actual code snippet from the repo and practical guidance.

## Metric Inventory

- Text metrics (single-run): WER, MER, WIL, CER, Levenshtein Distance, Normalized Levenshtein (`lev_norm`), Completeness
- Structural metrics (single-run): Heading Alignment, List Accuracy, Table Preservation, Link Correctness, Section Ordering
- Multi-run (batch) metrics: Aggregated Mean, Standard Deviation, Consistency Confidence Index (CCI), Variance Trend, Composite Scores (Text, Structural, Overall)

---

## Preprocessing Used by Metrics

- Purpose: Normalize unicode/whitespace and strip confounding characters prior to comparisons.
- Code:

```python
# ocr_eval_utils.py
import unicodedata
import re

def preprocess_markdown_for_evaluation(text):
    """
    Preprocess markdown text for fair comparison by normalizing formatting
    while preserving content structure.
    """
    if text is None:
        return ""
    try:
        text = unicodedata.normalize("NFC", text)
    except Exception:
        pass
    text = re.sub(r"[\u200B\u200C\u200D\uFEFF\u2060]", "", text)
    text = text.replace("\u00A0", " ").replace("\u202F", " ")
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    return text.strip()
```

- Notes: Applied inside `compute_all_metrics()` to both GT and OCR before lexical metrics.

---

## Word Error Rate (WER)

- Implementation: Uses JiWER after preprocessing.
- Code path: [ocr_eval_utils.py](ocr_eval_utils.py#L123-L145)
- Code:

```python
# ocr_eval_utils.py
import jiwer

def compute_all_metrics(gt_text: str, ocr_text: str):
    gt_processed = preprocess_markdown_for_evaluation(gt_text or "")
    ocr_processed = preprocess_markdown_for_evaluation(ocr_text or "")
    try:
        wer = jiwer.wer(gt_processed, ocr_processed)
        wer = float(max(0.0, min(1.0, float(wer))))  # clamp to [0,1]
        mer = jiwer.mer(gt_processed, ocr_processed)
        wil = jiwer.wil(gt_processed, ocr_processed)
    except Exception as e:
        print(f"Warning: Error computing word-level metrics: {e}")
        wer = mer = wil = 0.0
    # ... other metrics below
```

- Interpretation in framework: Lower is better; clamped to [0,1] to avoid large-insertion anomalies.

---

## Match Error Rate (MER)

- Implementation: JiWER’s MER alongside WER/WIL.
- Code path: [ocr_eval_utils.py](ocr_eval_utils.py#L123-L145)
- Code: see WER snippet above (`mer = jiwer.mer(...)`).
- Interpretation: Lower is better; denominator includes insertions for robustness against over-generation.

---

## Word Information Lost (WIL)

- Implementation: JiWER’s WIL complementing WIP (not directly exposed in framework).
- Code path: [ocr_eval_utils.py](ocr_eval_utils.py#L123-L145)
- Code: see WER snippet above (`wil = jiwer.wil(...)`).
- Interpretation: Lower is better; reflects information preservation relative to GT and hypothesis lengths.

---

## Character Error Rate (CER)

- Implementation: Custom CER helper using RapidFuzz Levenshtein with max-length normalization.
- Code path: [ocr_eval_utils.py](ocr_eval_utils.py#L36-L67)
- Code:

```python
# ocr_eval_utils.py
from rapidfuzz.distance import Levenshtein

def cer(gt, pred):
    gt_s = gt or ""
    pred_s = pred or ""
    denom = max(1, max(len(gt_s), len(pred_s)))
    try:
        dist = Levenshtein.distance(gt_s, pred_s)
    except Exception:
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, gt_s, pred_s).ratio()
        dist = int(round((1.0 - ratio) * denom))
    val = float(dist) / float(denom)
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val
```

- Use: `cer_value = cer(gt_processed, ocr_processed)` inside `compute_all_metrics()`.
- Interpretation: Lower is better; bounded [0,1].

---

## Levenshtein Distance and Normalized Levenshtein (`lev_norm`)

- Implementation: RapidFuzz character distance; normalized by max length.
- Code path: [ocr_eval_utils.py](ocr_eval_utils.py#L170-L191)
- Code:

```python
# ocr_eval_utils.py
from rapidfuzz.distance import Levenshtein

def compute_all_metrics(gt_text: str, ocr_text: str):
    # ... after preprocessing
    try:
        gt_str = str(gt_processed) if gt_processed else ""
        ocr_str = str(ocr_processed) if ocr_processed else ""
        lev_distance = int(Levenshtein.distance(gt_str, ocr_str))
        lev_norm_denom = max(1, max(len(gt_str), len(ocr_str)))
        lev_norm = float(lev_distance) / float(lev_norm_denom)
    except Exception as e:
        print(f"Warning: Error computing Levenshtein distance: {e}")
        lev_distance = 0
        lev_norm = 0.0
```

- Interpretation: Lower is better; `lev_norm` enables fair cross-document comparison.

---

## Completeness

- Implementation: Ratio of preprocessed OCR length to GT length, capped at 1.
- Code path: [ocr_eval_utils.py](ocr_eval_utils.py#L192-L216)
- Code:

```python
# ocr_eval_utils.py

def compute_all_metrics(gt_text: str, ocr_text: str):
    # ... after structural analysis
    try:
        ocr_len = int(len(str(ocr_processed)))
        gt_len = int(len(str(gt_processed)))
        completeness = float(min(1.0, ocr_len / max(1, gt_len)))
    except Exception as e:
        print(f"Warning: Error computing completeness: {e}")
        completeness = 0.0
```

- Interpretation: 1.0 ideal; <1 indicates truncation/missing text (verbosity is not penalized here).

---

## Structural Analysis Metrics

- Implementation: Unified analyzer over raw markdown (not preprocessed), producing structure counts and accuracy metrics.
- Code path: [ocr_eval_utils.py](ocr_eval_utils.py#L100-L241)
- Code:

```python
# ocr_eval_utils.py
import difflib
import re
import unicodedata

def analyze_markdown_structure(gt_text: str, ocr_text: str):
    gt_text = str(gt_text) if gt_text is not None else ""
    ocr_text = str(ocr_text) if ocr_text is not None else ""

    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKC", s or "")
        s = s.lower()
        s = re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', s)).strip()
        return s

    header_pattern = re.compile(r'^(#{1,6})\s*(.+)$', re.MULTILINE)
    gt_headers_matches = header_pattern.findall(gt_text or "")
    ocr_headers_matches = header_pattern.findall(ocr_text or "")
    gt_header_texts = [h[1].strip() for h in gt_headers_matches]
    ocr_header_texts = [h[1].strip() for h in ocr_headers_matches]
    gt_hdr_norm = [_norm(h) for h in gt_header_texts]
    ocr_hdr_norm = [_norm(h) for h in ocr_header_texts]

    table_line_pattern = re.compile(r'\|.*\|')
    gt_table_lines = table_line_pattern.findall(gt_text or "")
    ocr_table_lines = table_line_pattern.findall(ocr_text or "")
    gt_tables = len(gt_table_lines)
    ocr_tables = len(ocr_table_lines)

    list_bullet = re.compile(r'^[\*\-\+]\s', re.MULTILINE)
    list_ordered = re.compile(r'^\d+\.\s', re.MULTILINE)
    gt_lists = len(list_bullet.findall(gt_text or "")) + len(list_ordered.findall(gt_text or ""))
    ocr_lists = len(list_bullet.findall(ocr_text or "")) + len(list_ordered.findall(ocr_text or ""))

    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    gt_links = link_pattern.findall(gt_text or "")
    ocr_links = link_pattern.findall(ocr_text or "")
    gt_targets = [t[1].strip() for t in gt_links]
    ocr_targets = [t[1].strip() for t in ocr_links]
    gt_targets_norm = set(_norm(t) for t in gt_targets if _norm(t))
    ocr_targets_norm = set(_norm(t) for t in ocr_targets if _norm(t))

    heading_alignment = 1.0
    section_ordering = 1.0
    if gt_hdr_norm or ocr_hdr_norm:
        THRESH = 0.80
        matches = []
        j_start = 0
        for i, g in enumerate(gt_hdr_norm):
            best_j = -1
            best_sim = 0.0
            for j in range(j_start, len(ocr_hdr_norm)):
                sim = difflib.SequenceMatcher(None, g, ocr_hdr_norm[j]).ratio()
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            if best_j >= j_start and best_sim >= THRESH:
                matches.append((i, best_j, best_sim))
                j_start = best_j + 1
        if not matches:
            heading_alignment = 0.0
            section_ordering = 1.0 if (len(gt_hdr_norm) <= 1 and len(ocr_hdr_norm) <= 1) else 0.5
        else:
            avg_sim = sum(m[2] for m in matches) / len(matches)
            coverage = len(matches) / max(1, max(len(gt_hdr_norm), len(ocr_hdr_norm)))
            heading_alignment = max(0.0, min(1.0, avg_sim * coverage))
            ocr_index_seq = [m[1] for m in matches]
            def _lcs_len(seq):
                n = len(seq)
                if n <= 1:
                    return n
                if all(seq[k] < seq[k+1] for k in range(n-1)):
                    return n
                tails = []
                import bisect
                for x in seq:
                    i = bisect.bisect_left(tails, x)
                    if i == len(tails):
                        tails.append(x)
                    else:
                        tails[i] = x
                return len(tails)
            lcs = _lcs_len(ocr_index_seq)
            section_ordering = max(0.0, min(1.0, (lcs / max(1, len(matches))) * coverage))

    if gt_lists == 0 and ocr_lists == 0:
        list_accuracy = 1.0
    else:
        list_accuracy = 1.0 - (abs(gt_lists - ocr_lists) / max(1, max(gt_lists, ocr_lists)))
        list_accuracy = max(0.0, min(1.0, list_accuracy))

    def _estimate_columns_from_lines(lines):
        cols = []
        for ln in lines:
            parts = [p for p in ln.strip().split('|') if p.strip() != ""]
            if len(parts) >= 2:
                cols.append(len(parts))
        return cols

    def _extract_table_content(lines):
        content = set()
        for ln in lines:
            parts = [_norm(p) for p in ln.strip().split('|') if p.strip() != ""]
            content.update(p for p in parts if p)
        return content

    if gt_tables == 0 and ocr_tables == 0:
        table_preservation = None
    else:
        presence = min(gt_tables, ocr_tables) / max(1, max(gt_tables, ocr_tables))
        gt_cols_list = _estimate_columns_from_lines(gt_table_lines)
        ocr_cols_list = _estimate_columns_from_lines(ocr_table_lines)
        col_score = 0.0
        if gt_cols_list and ocr_cols_list:
            pairs = min(len(gt_cols_list), len(ocr_cols_list))
            sims = []
            for i in range(pairs):
                a, b = gt_cols_list[i], ocr_cols_list[i]
                sims.append(min(a, b) / max(1, max(a, b)))
            col_score = sum(sims) / len(sims) if sims else 0.0
        gt_content = _extract_table_content(gt_table_lines)
        ocr_content = _extract_table_content(ocr_table_lines)
        if gt_content or ocr_content:
            union = gt_content | ocr_content
            inter = gt_content & ocr_content
            content_score = len(inter) / max(1, len(union))
        else:
            content_score = 1.0
        table_preservation = max(0.0, min(1.0, 0.4 * presence + 0.3 * col_score + 0.3 * content_score))

    if not gt_targets_norm and not ocr_targets_norm:
        link_correctness = 1.0
    else:
        union = gt_targets_norm | ocr_targets_norm
        inter = gt_targets_norm & ocr_targets_norm
        link_correctness = len(inter) / max(1, len(union))

    structure_metrics = {
        'headers': {
            'gt': len(gt_header_texts),
            'ocr': len(ocr_header_texts),
            'gt_texts': [h.lower() for h in gt_header_texts],
            'ocr_texts': [h.lower() for h in ocr_header_texts],
        },
        'tables': {
            'gt': gt_tables,
            'ocr': ocr_tables,
            'gt_table_lines': gt_table_lines,
            'ocr_table_lines': ocr_table_lines,
        },
        'lists': {
            'gt': gt_lists,
            'ocr': ocr_lists,
        },
        'links': {
            'gt': len(gt_targets),
            'ocr': len(ocr_targets),
            'gt_targets': gt_targets,
            'ocr_targets': ocr_targets,
        }
    }

    structural_analysis = {
        'heading_alignment': float(max(0.0, min(1.0, heading_alignment))),
        'list_accuracy': float(max(0.0, min(1.0, list_accuracy))),
        'table_preservation': None if table_preservation is None else float(max(0.0, min(1.0, table_preservation))),
        'link_correctness': float(max(0.0, min(1.0, link_correctness))),
        'section_ordering': float(max(0.0, min(1.0, section_ordering))),
    }

    return structure_metrics, structural_analysis
```

- Interpretation: Structural metrics in [0,1] (higher better), except `table_preservation` may be NA when tables absent.

---

## Top Word Mismatches (Diagnostic)

- Implementation: Diff-based extraction of replacements/deletions/insertions with frequency.
- Code path: [ocr_eval_utils.py](ocr_eval_utils.py#L263-L303)
- Code:

```python
# ocr_eval_utils.py
from collections import Counter
import difflib

def compute_top_word_mismatches(gt_text, ocr_text, top_n=10):
    gt_words = gt_text.split()
    ocr_words = ocr_text.split()
    matcher = difflib.SequenceMatcher(None, gt_words, ocr_words)
    mismatches = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            for g, o in zip(gt_words[i1:i2], ocr_words[j1:j2]):
                mismatches.append((g, o))
        elif tag == "delete":
            for g in gt_words[i1:i2]:
                mismatches.append((g, "<MISSING>"))
        elif tag == "insert":
            for o in ocr_words[j1:j2]:
                mismatches.append(("<EXTRA>", o))
    counter = Counter(mismatches)
    return counter.most_common(top_n)
```

- Interpretation: Useful for QA triage; not a score.

---

## Multi-Run Aggregation: Mean, Std Dev, CCI, Variance Trend

- Implementation: Aggregates per metric across runs, computes CCI and trend.
- Code path: [multi_run_evaluation.py](multi_run_evaluation.py#L81-L170)
- Code:

```python
# multi_run_evaluation.py
import statistics
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class AggregatedMetric:
    mean: float
    std_dev: float
    cci: float
    variance_trend: str
    values: List[float] = None
    def __post_init__(self):
        if self.values is None:
            self.values = []

class MultiRunAggregator:
    @staticmethod
    def compute_aggregated_metric(values: List[float]) -> AggregatedMetric:
        valid_values: List[float] = []
        for v in values:
            if v is None:
                continue
            try:
                valid_values.append(float(v))
            except Exception:
                continue
        if not valid_values:
            return AggregatedMetric(None, None, None, "not_applicable", values)
        mean_val = float(statistics.mean(valid_values))
        std_dev = float(statistics.stdev(valid_values)) if len(valid_values) > 1 else 0.0
        if float(mean_val) > 0:
            cci = max(0.0, 1.0 - (std_dev / mean_val))
        else:
            cci = 0.0
        if len(valid_values) < 3:
            variance_trend = "insufficient_data"
        else:
            first_half = valid_values[:len(valid_values)//2]
            second_half = valid_values[len(valid_values)//2:]
            first_mean = statistics.mean(first_half)
            second_mean = statistics.mean(second_half)
            diff_ratio = abs(second_mean - first_mean) / max(first_mean, 0.001)
            if diff_ratio < 0.05:
                variance_trend = "stable"
            elif second_mean > first_mean:
                variance_trend = "increasing"
            else:
                variance_trend = "decreasing"
        return AggregatedMetric(mean_val, std_dev, cci, variance_trend, values.copy())

    @staticmethod
    def compute_overall_cci(aggregated_metrics: Dict[str, AggregatedMetric]) -> float:
        cci_values = [m.cci for m in aggregated_metrics.values() if m.cci is not None and m.cci > 0]
        return statistics.mean(cci_values) if cci_values else 0.0

    @staticmethod
    def interpret_stability(overall_cci: float) -> str:
        if overall_cci > 0.90:
            return "Highly stable"
        elif overall_cci >= 0.75:
            return "Moderately stable"
        else:
            return "Unstable results"
```

- Interpretation thresholds: Reporter uses these bands and emojis when rendering.

---

## Multi-Run Composite Scores (Text, Structural, Overall)

- Implementation: Derived from aggregated metric means.
- Code path: [multi_run_evaluation.py](multi_run_evaluation.py#L171-L233) and reporter backfill [multi_run_reporter.py](multi_run_reporter.py#L116-L171)
- Code:

```python
# multi_run_evaluation.py
import statistics
from typing import Optional, List, Dict

class MultiRunAggregator:
    @staticmethod
    def compute_composite_scores(aggregated_metrics: Dict[str, AggregatedMetric]) -> Dict[str, Optional[float]]:
        def _collect_means(names: List[str]) -> List[float]:
            vals: List[float] = []
            for n in names:
                m = aggregated_metrics.get(n)
                if m is not None and m.mean is not None:
                    try:
                        vals.append(float(m.mean))
                    except Exception:
                        continue
            return vals
        text_components = _collect_means(['wer', 'cer', 'mer', 'wil'])
        if text_components:
            avg_err = statistics.mean(text_components)
            text_score = max(0.0, min(1.0, 1.0 - float(avg_err)))
        else:
            text_score = None
        structural_components = _collect_means([
            'heading_alignment', 'list_accuracy', 'table_preservation', 'link_correctness', 'section_ordering'
        ])
        if structural_components:
            structural_score = max(0.0, min(1.0, float(statistics.mean(structural_components))))
        else:
            structural_score = None
        if text_score is not None and structural_score is not None:
            overall_score = max(0.0, min(1.0, 0.6 * text_score + 0.4 * structural_score))
        else:
            overall_score = None
        return {
            'text_score': text_score,
            'structural_score': structural_score,
            'overall_score': overall_score,
        }
```

- Interpretation: Text score converts average error to accuracy; structural score averages structural metrics; overall blends them 60/40.

---

## Reporter Interpretation of CCI and Display

- Implementation: Emoji and thresholds for aggregated metrics table.
- Code path: [multi_run_reporter.py](multi_run_reporter.py#L40-L115)
- Code:

```python
# multi_run_reporter.py
    def generate_aggregated_metrics_table(self, aggregated_metrics: Dict[str, AggregatedMetric]) -> str:
        lines = []
        lines.append("| Metric | Mean | Std Dev | CCI | Variance Trend | Interpretation |")
        for metric_name, agg_metric in aggregated_metrics.items():
            cci_val = agg_metric.cci
            if cci_val is None:
                cci_interp = "NA"
            elif cci_val > 0.90:
                cci_interp = "🟢 Highly Stable"
            elif cci_val >= 0.75:
                cci_interp = "🟡 Moderately Stable"
            else:
                cci_interp = "🔴 Unstable"
            trend_emoji = {
                "stable": "➡️",
                "increasing": "📈",
                "decreasing": "📉",
                "insufficient_data": "❓",
                "not_applicable": "NA"
            }
            trend_display = f"{trend_emoji.get(agg_metric.variance_trend, '❓')} {agg_metric.variance_trend.replace('_', ' ').title() if agg_metric.variance_trend else 'NA'}"
            lines.append(
                f"| **{metric_name.upper()}** | "
                f"{self._fmt(agg_metric.mean)} | "
                f"{self._fmt(agg_metric.std_dev)} | "
                f"{self._fmt(agg_metric.cci)} | "
                f"{trend_display} | "
                f"{cci_interp} |"
            )
        return "\n".join(lines) + "\n"
```

---

## How Metrics Work Together

- Lexical similarity: WER, MER, WIL, CER, Levenshtein, `lev_norm` (lower better)
- Structural adherence: Heading Alignment, List Accuracy, Table Preservation, Link Correctness, Section Ordering (higher better)
- Batch stability & trajectory: Mean, Std Dev, CCI, Variance Trend
- Composite: Text Accuracy Score (1 − avg errors), Structural Score (avg structural metrics), Overall (0.6×Text + 0.4×Structural)

---

## Practical Recommendations

- Pair WER with CER and `lev_norm` for robust lexical accuracy.
- Always show mean + std dev + CCI in multi-run to communicate reliability.
- Use Structural Score to ensure format fidelity; Overall score for product dashboards.
- Normalize texts with `preprocess_markdown_for_evaluation()` to avoid spurious mismatches.
