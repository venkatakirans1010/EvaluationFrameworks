# OCR Evaluation Framework Metrics Documentation

This document is the official reference for metrics used in this OCR/LLM evaluation framework. It merges theory, math, framework-specific implementation details, preprocessing, aggregation, reporter interpretation, and practical guidance. It is intended for ML Engineers, QA Engineers, Data Scientists, and Product stakeholders.

## Metric Inventory

- Text similarity and error metrics (single-run):
  - Word Error Rate (WER)
  - Match Error Rate (MER)
  - Word Information Lost (WIL)
  - Character Error Rate (CER)
  - Levenshtein Distance (characters)
  - Normalized Levenshtein Distance (`lev_norm`)
  - Completeness
- Structural metrics (single-run):
  - Heading Alignment
  - List Accuracy
  - Table Preservation
  - Link Correctness
  - Section Ordering
- Multi-run aggregate metrics (batch):
  - Aggregated Mean
  - Standard Deviation
  - Consistency Confidence Index (CCI)
  - Variance Trend
  - Composite Scores: Text Accuracy Score, Structural Score, Overall Extraction Score

---

## Preprocessing Used by Metrics

- Purpose: Normalize unicode/whitespace and strip confounding characters prior to comparisons.
- Implementation: Applied inside `compute_all_metrics()` to both GT and OCR before lexical metrics.
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

---

## Word Error Rate (WER)

- What This Measures: Lexical accuracy at word level; measures edits needed to transform OCR/LLM output into GT.
- When to Use: OCR/ASR outputs compared to a reference transcript; QA for text extraction.
- Math: WER = (S + D + I) / N, where N is GT word count.
- Framework Implementation: Uses JiWER after preprocessing; clamped to [0,1] to avoid extreme-insertion anomalies.
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

- Example: GT “hello world”, Hyp “hello duck” → WER = (1+0+0)/2 = 0.5.
- Strengths: Simple, standard, sensitive to insertions/deletions/substitutions.
- Limitations: Penalizes benign rephrasings; ignores semantics.
- Practical Tip: Use with CER and `lev_norm` to balance word-level and character-level sensitivity.

---

## Match Error Rate (MER)

- What This Measures: Proportion of word-level errors normalized by total aligned tokens (accounts for insertions in denominator).
- When to Use: Scenarios with extra words; noisy outputs.
- Math: MER = (S + D + I) / (C + S + D + I) where C=hits.
- Framework Implementation: `jiwer.mer` in `compute_all_metrics()`.
- Code path: [ocr_eval_utils.py](ocr_eval_utils.py#L123-L145)
- Example: GT “a b c”, Hyp “a x c y” → MER = (1+0+1)/(2+1+0+1) = 0.5.
- Strengths: Penalizes insertions; stable under over-generation.
- Limitations: Still lexical; not semantic.
- Practical Tip: Use MER alongside WER to diagnose insertion-heavy outputs.

---

## Word Information Lost (WIL)

- What This Measures: Fraction of word information lost; complements WIP (information preserved).
- When to Use: Quality reporting where information preservation matters.
- Math: WIP = (C/N) × (C/M); WIL = 1 − WIP.
- Framework Implementation: `jiwer.wil` in `compute_all_metrics()`.
- Code path: [ocr_eval_utils.py](ocr_eval_utils.py#L123-L145)
- Example: GT “hello world”, Hyp “hello duck” → WIL = 0.75.
- Strengths: Accounts for verbosity and missing content.
- Limitations: Sensitive to length ratios; may be unintuitive.
- Practical Tip: Include with WER/MER to reflect information preservation.

---

## Character Error Rate (CER)

- What This Measures: Lexical accuracy at character level; Levenshtein-based.
- When to Use: OCR of scanned text; languages with complex scripts.
- Math: CER = LevenshteinDistance(gt, hyp) / max(|gt|, |hyp|); bounded to [0,1].
- Framework Implementation: Custom `cer()` helper using RapidFuzz.
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

- Example: GT “abcde”, Hyp “abxde” → distance=1, denom=5 → CER=0.2.
- Strengths: Captures fine-grained OCR errors.
- Limitations: Over-penalizes harmless character differences; ignores semantics.
- Practical Tip: Normalize Unicode and whitespace to avoid spurious mismatches.

---

## Levenshtein Distance and Normalized Levenshtein (`lev_norm`)

- What These Measure: Raw character edit distance and its normalized variant for cross-document comparability.
- When to Use: Debugging (`distance`), aggregated reporting (`lev_norm`).
- Math: `lev_norm = distance / max(1, max(len(gt), len(hyp)))`.
- Framework Implementation: RapidFuzz character distance; normalized by max length.
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

- Example: GT “abc”, Hyp “abx” → distance=1, denom=3 → `lev_norm`=0.3333.
- Strengths: Fair across lengths; intuitive.
- Limitations: Ignores semantics; sensitive to normalization.
- Practical Tip: Combine `distance` with `lev_norm` for fair comparisons.

---

## Completeness

- What This Measures: Fraction of extracted text length relative to ground truth; detects truncation or missing content.
- When to Use: Any extraction task where coverage matters.
- Math: completeness = min(1.0, |hyp| / max(1, |gt|)).
- Framework Implementation: Computed in `compute_all_metrics()` after structural analysis.
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

- Example: GT length=1000, Hyp length=800 → 0.8.
- Strengths: Simple coverage measure; catches truncation.
- Limitations: Ignores correctness; can be 1.0 with hallucinations.
- Practical Tip: Use alongside accuracy metrics to avoid false positives.

---

## Structural Analysis Metrics

Structural metrics operate on raw markdown (not the preprocessed text used for lexical metrics) and return values in [0,1] (higher is better), except `table_preservation` which may be `None` when tables are absent in both GT and output.

- Code path: [ocr_eval_utils.py](ocr_eval_utils.py#L100-L241)
- Core Implementation:

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

    # ... heading_alignment, list_accuracy, table_preservation, link_correctness, section_ordering
    # returns both structure_metrics and structural_analysis
```

### Heading Alignment

- Measures: Preservation and alignment of markdown headings using fuzzy matching.
- How: Greedy ordered matching with threshold; score = avg_similarity × coverage; clamp [0,1].
- Code path: [ocr_eval_utils.py](ocr_eval_utils.py#L100-L168)
- Example: GT [“Introduction”, “Methods”]; Hyp [“Intro”, “Methodology”] → score ~0.8–1.0.
- Strengths: Captures presence and quality; order-aware.
- Limitations: Heuristic threshold; sensitive to normalization.
- Tip: Monitor with Section Ordering.

### List Accuracy

- Measures: Preservation of list markers (# of bullets/ordered items).
- How: Score = 1 − |gt_lists − hyp_lists| / max(1, max(gt_lists, hyp_lists)); clamp [0,1].
- Code path: [ocr_eval_utils.py](ocr_eval_utils.py#L169-L216)
- Example: GT 10 bullets; Hyp 8 → 0.8.

### Table Preservation

- Measures: Presence, column structure similarity, and cell content overlap in markdown tables.
- How: Weighted composite: 0.4×presence + 0.3×col_score + 0.3×content_score; clamp [0,1]; `None` if both have no tables.
- Code path: [ocr_eval_utils.py](ocr_eval_utils.py#L187-L262)
- Example: Good column and content overlap → score ~0.6–0.8.

### Link Correctness

- Measures: Preservation of link targets; Jaccard overlap of normalized targets.
- How: Score = |intersection| / max(1, |union|).
- Code path: [ocr_eval_utils.py](ocr_eval_utils.py#L216-L241)
- Example: GT {a,b,c}, Hyp {a,c,d} → 0.5.

### Section Ordering

- Measures: Preservation of section order implied by matched headings.
- How: Approximate LCS/LIS length over matched indices; scales by coverage; clamp [0,1].
- Code path: [ocr_eval_utils.py](ocr_eval_utils.py#L140-L186)
- Example: Strictly increasing indices → high score (e.g., 0.9).

---

## Diagnostic: Top Word Mismatches

- Purpose: Identify frequent replacements/insertions/deletions to guide fixes.
- Implementation: Diff-based extraction with frequency.
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

- Note: Diagnostic only; not a score.

---

## Multi-Run Aggregation: Mean, Std Dev, CCI, Variance Trend

Aggregates per metric across runs, computes stability (CCI) and trajectory (Variance Trend).

- Code path: [multi_run_evaluation.py](multi_run_evaluation.py#L81-L170)
- Core Implementation:

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
- Examples:
  - Mean: WER runs [0.2, 0.25, 0.3] → mean=0.25.
  - Std Dev: same runs → std≈0.05.
  - CCI: 1 − (0.05/0.25) = 0.8 → moderately stable.
  - Trend: [0.30, 0.28, 0.25] → decreasing.

---

## Multi-Run Composite Scores (Text, Structural, Overall)

- Code path: [multi_run_evaluation.py](multi_run_evaluation.py#L171-L233) and reporter backfill [multi_run_reporter.py](multi_run_reporter.py#L116-L171)
- Implementation:

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

- Examples:
  - Text: WER=0.2, CER=0.1, MER=0.15, WIL=0.18 → avg_err=0.1575 → score=0.8425.
  - Structural: [0.9, 0.8, 0.7, 0.9, 0.85] → score=0.83.
  - Overall: 0.6×0.8425 + 0.4×0.83 ≈ 0.838.

---

## Reporter Interpretation of CCI and Display

- Purpose: Emoji and thresholds for aggregated metrics table.
- Code path: [multi_run_reporter.py](multi_run_reporter.py#L40-L115)
- Implementation:

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

- Interpretation Bands:
  - 🟢 CCI > 0.90: Highly Stable
  - 🟡 0.75 ≤ CCI ≤ 0.90: Moderately Stable
  - 🔴 CCI < 0.75: Unstable
  - Trend: ➡️ stable, 📈 increasing, 📉 decreasing, ❓ insufficient data

---

## How Metrics Work Together

- Lexical similarity (lower better): WER, MER, WIL, CER, Levenshtein, `lev_norm`.
- Structural adherence (higher better): Heading Alignment, List Accuracy, Table Preservation, Link Correctness, Section Ordering.
- Batch stability & trajectory: Mean, Std Dev, CCI, Variance Trend.
- Composite: Text Accuracy Score (1 − avg errors), Structural Score (avg structural metrics), Overall (0.6×Text + 0.4×Structural).

---

## Practical Recommendations

- Pair WER with CER and `lev_norm` for robust lexical accuracy.
- Always show mean + std dev + CCI in multi-run to communicate reliability.
- Use Structural Score to ensure format fidelity; Overall score for product dashboards.
- Normalize texts with `preprocess_markdown_for_evaluation()` to avoid spurious mismatches.
- Treat Completeness as coverage only; combine with accuracy metrics.
- Monitor Heading Alignment with Section Ordering to catch reordering.

---

## References

- JiWER library: https://jitsi.github.io/jiwer/ and https://github.com/jitsi/jiwer
- RapidFuzz: https://github.com/maxbachmann/RapidFuzz