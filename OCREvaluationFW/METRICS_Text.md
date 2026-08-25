# OCR Evaluation Framework Metrics Documentation

This document describes every metric used in the framework for single-run and multi-run (batch) evaluation. It is intended for ML Engineers, QA Engineers, Data Scientists, and Product stakeholders. Each metric includes what it measures, when to use it, how it works, mathematical definitions, implementation references, example calculations, strengths/limitations, pitfalls, performance notes, tuning, related metrics, and practical recommendations.

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

## Word Error Rate (WER)

- What This Metric Measures:
  - Aspect: Lexical accuracy at word level; measures how many word edits are needed to transform the OCR output into the ground truth.
  - Importance: Widely used, intuitive, and sensitive to insertions, deletions, and substitutions.
- When to Use This Metric:
  - Best: OCR, ASR outputs compared to a reference transcript; QA for text extraction; general text accuracy.
  - Not reliable: Highly paraphrased outputs, free-form summarization, translation across languages, or tasks prioritizing semantics over exact wording.
- How It Works (Conceptual):
  - Compares aligned word sequences between reference and hypothesis; counts substitutions (S), deletions (D), and insertions (I) versus reference word count (N).
  - Good vs Bad: Lower is better; 0.0 means perfect match, 1.0 means completely wrong (or extreme insertions for empty reference).
- Mathematical / Algorithmic:
  - Formula: WER = (S + D + I) / N, where N = number of words in reference.
  - Implementation uses RapidFuzz alignment via `jiwer.process_words()`.
- Code Implementation (From Framework):
  - Computed in `compute_all_metrics()` using `jiwer.wer` in [ocr_eval_utils.py](ocr_eval_utils.py#L123-L145).
- Example Calculation:
  - Ground Truth: “hello world”
  - LLM Output: “hello duck”
  - Alignment: hits=1, S=1 (world→duck), D=0, I=0, N=2
  - Step-by-Step: WER = (1+0+0)/2 = 0.5
  - Final Score: 0.5
- Strengths:
  - Simple, standard, sensitive to all edit types.
- Limitations:
  - Penalizes benign rephrasings; ignores semantics; casing/punctuation can impact score unless normalized.
- Common Pitfalls:
  - Not normalizing text causes inflated errors; comparing paraphrases; using WER solo for semantic tasks.
- Performance Considerations:
  - Fast; suitable for real-time or batch evaluation.
- Tuning & Configuration:
  - Preprocessing and normalization (whitespace, case); optionally ignore image descriptions (see `evaluate_ocr_performance()` flag) in [compare.py](compare.py#L22-L70).
- Related Metrics:
  - MER, WIL, CER, Levenshtein.
- Practical Recommendation:
  - Use with `CER` and `lev_norm` to balance word-level and character-level sensitivity.

---

## Match Error Rate (MER)

- What This Metric Measures:
  - Aspect: Overall proportion of word-level errors normalized by the total alignment length including insertions.
  - Importance: Accounts for insertions in denominator; robust in scenarios with extra words.
- When to Use This Metric:
  - Best: OCR/ASR where insertions are common; noisy outputs; data cleaning.
  - Not reliable: Semantic tasks; when reference is extremely short or empty (edge-case handling applies).
- How It Works (Conceptual):
  - Uses word-level alignment; counts S, D, I and normalizes by total aligned tokens (hits+S+D+I).
  - Good vs Bad: Lower is better; penalizes over-generation via denominator.
- Mathematical / Algorithmic:
  - Formula (from jiwer): MER = (S + D + I) / (C + S + D + I) where `C`=hits.
- Code Implementation (From Framework):
  - Computed via `jiwer.mer` in `compute_all_metrics()` in [ocr_eval_utils.py](ocr_eval_utils.py#L123-L145).
  - JiWER reference implementation: see `WordOutput.mer` in upstream [jiwer process code](https://raw.githubusercontent.com/jitsi/jiwer/master/src/jiwer/process.py) (MER formula around returned `output.mer`).
- Example Calculation:
  - Ground Truth: “a b c”
  - LLM Output: “a x c y” → hits=2, S=1, D=0, I=1
  - MER = (1+0+1)/(2+1+0+1) = 2/4 = 0.5
  - Final Score: 0.5
- Strengths:
  - Penalizes insertions; more stable under over-generated content.
- Limitations:
  - Still lexical; not semantic; can under-represent severity when many insertions are harmless.
- Common Pitfalls:
  - Misinterpreting MER as semantic quality; ignoring normalization differences.
- Performance Considerations:
  - Fast; same complexity as WER; suitable for batch.
- Tuning & Configuration:
  - Align preprocessing with WER; consistent tokenization.
- Related Metrics:
  - WIL (complement with information perspective), WER.
- Practical Recommendation:
  - Use MER alongside WER to diagnose insertion-heavy outputs.

---

## Word Information Lost (WIL)

- What This Metric Measures:
  - Aspect: Fraction of word information lost; complements `WIP` (information preserved).
  - Importance: Captures both recognition accuracy and verbosity effects.
- When to Use This Metric:
  - Best: OCR/ASR quality reporting where information preservation matters.
  - Not reliable: Semantic paraphrases; summarization.
- How It Works (Conceptual):
  - Derived from `WIP = (hits / ref_words) * (hits / hyp_words)`; `WIL = 1 - WIP`.
  - Good vs Bad: Lower WIL is better; 0 means all information preserved.
- Mathematical / Algorithmic:
  - WIP = (C / N) × (C / M); WIL = 1 − WIP.
  - C=hits, N=reference words, M=hypothesis words.
- Code Implementation (From Framework):
  - Computed via `jiwer.wil` in `compute_all_metrics()` in [ocr_eval_utils.py](ocr_eval_utils.py#L123-L145).
  - Upstream calculation visible in [jiwer process code](https://raw.githubusercontent.com/jitsi/jiwer/master/src/jiwer/process.py) (`wip` and `wil`).
- Example Calculation:
  - GT: “hello world”, Hyp: “hello duck” → C=1, N=2, M=2
  - WIP = (1/2)×(1/2)=0.25 → WIL = 1−0.25 = 0.75
  - Final Score: 0.75
- Strengths:
  - Accounts for verbosity and missing content.
- Limitations:
  - Sensitive to reference/hyp length ratios; may be unintuitive.
- Common Pitfalls:
  - Comparing WIL across documents with drastically different lengths.
- Performance Considerations:
  - Fast; same as WER/MER.
- Tuning & Configuration:
  - Ensure consistent normalization; avoid counting non-content tokens.
- Related Metrics:
  - WER, MER, WIP (not reported here but implied).
- Practical Recommendation:
  - Include WIL with WER/MER to reflect information preservation, especially for long texts.

---

## Character Error Rate (CER)

- What This Metric Measures:
  - Aspect: Lexical accuracy at character level; Levenshtein-based.
  - Importance: Sensitive to minor OCR noise like diacritics, punctuation, or tight spacing.
- When to Use This Metric:
  - Best: OCR of scanned text; languages with complex scripts; noisy image text.
  - Not reliable: Pure semantic tasks; when tokenization quality matters more than characters.
- How It Works (Conceptual):
  - Compares characters using Levenshtein distance, normalized to [0,1] using max length of GT or hypothesis.
  - Good vs Bad: Lower is better.
- Mathematical / Algorithmic:
  - Our framework’s CER: CER = LevenshteinDistance(gt, hyp) / max(|gt|, |hyp|).
  - Bounds enforced to [0,1]; RapidFuzz used for distance.
- Code Implementation (From Framework):
  - `cer()` helper in [ocr_eval_utils.py](ocr_eval_utils.py#L36-L67).
  - Used in `compute_all_metrics()` in [ocr_eval_utils.py](ocr_eval_utils.py#L146-L169).
- Example Calculation:
  - GT: “abcde”, Hyp: “abxde” → distance=1, denom=max(5,5)=5 → CER=0.2
  - Final Score: 0.2
- Strengths:
  - Captures fine-grained OCR errors; language-agnostic.
- Limitations:
  - Over-penalizes harmless character differences; insensitive to semantics.
- Common Pitfalls:
  - Not normalizing Unicode; counting zero-width marks; overlooking NBSP.
- Performance Considerations:
  - Very fast; suitable for large batch.
- Tuning & Configuration:
  - Unicode normalization, whitespace collapse; framework `preprocess_markdown_for_evaluation()` in [ocr_eval_utils.py](ocr_eval_utils.py#L9-L33, ocr_eval_utils.py#L69-L98).
- Related Metrics:
  - WER/MER/WIL (word-level), `lev_norm`.
- Practical Recommendation:
  - Use CER alongside WER to diagnose small OCR glitches.

---

## Levenshtein Distance (characters)

- What This Metric Measures:
  - Aspect: Raw character edit distance (absolute count of edits).
  - Importance: Diagnostic visibility for absolute error magnitude.
- When to Use This Metric:
  - Best: Debugging; inspecting changes; non-normalized scale helpful.
  - Not reliable: Cross-document comparisons (length bias).
- How It Works (Conceptual):
  - Counts minimal edits between GT and hypothesis.
  - Good vs Bad: Lower is better; 0 means identical strings.
- Mathematical / Algorithmic:
  - Standard Levenshtein distance over characters.
- Code Implementation (From Framework):
  - Computed via RapidFuzz in `compute_all_metrics()` in [ocr_eval_utils.py](ocr_eval_utils.py#L170-L191).
- Example Calculation:
  - GT: “kitten”, Hyp: “sitting” → distance=3.
  - Final Score: 3
- Strengths:
  - Simple; language-agnostic; useful for absolute comparison.
- Limitations:
  - Length-dependent; not bounded.
- Common Pitfalls:
  - Interpreting distances without normalization.
- Performance Considerations:
  - Fast; efficient through RapidFuzz.
- Tuning & Configuration:
  - Apply normalization before measuring to avoid spurious distances.
- Related Metrics:
  - `lev_norm`, CER.
- Practical Recommendation:
  - Combine with `lev_norm` to enable fair comparisons across varying lengths.

---

## Normalized Levenshtein Distance (`lev_norm`)

- What This Metric Measures:
  - Aspect: Character distance normalized to [0,1] by max length.
  - Importance: Enables cross-document comparability.
- When to Use This Metric:
  - Best: Aggregated reporting; comparing documents of different sizes.
  - Not reliable: Extreme empty references or outputs (edge-case handling applies).
- How It Works (Conceptual):
  - Same as Levenshtein but divided by max(|gt|,|hyp|).
  - Good vs Bad: Lower is better.
- Mathematical / Algorithmic:
  - `lev_norm = distance / max(1, max(len(gt), len(hyp)))`.
- Code Implementation (From Framework):
  - See `compute_all_metrics()` in [ocr_eval_utils.py](ocr_eval_utils.py#L170-L191).
- Example Calculation:
  - GT: “abc”, Hyp: “abx” → distance=1, denom=3 → `lev_norm`=0.3333.
  - Final Score: 0.3333
- Strengths:
  - Fair across lengths; intuitive.
- Limitations:
  - Still lexical; ignores semantics; sensitive to normalization.
- Common Pitfalls:
  - Not normalizing Unicode and whitespace increases distance.
- Performance Considerations:
  - Fast.
- Tuning & Configuration:
  - Same normalization suggestions as CER.
- Related Metrics:
  - CER; WER.
- Practical Recommendation:
  - Include `lev_norm` in composite text accuracy scoring for stability.

---

## Completeness

- What This Metric Measures:
  - Aspect: Fraction of extracted text length relative to ground truth.
  - Importance: Detects truncation or missing content in OCR/LLM outputs.
- When to Use This Metric:
  - Best: Any extraction task where coverage matters; pipeline regression checks.
  - Not reliable: Overly verbose outputs; doesn’t capture correctness.
- How It Works (Conceptual):
  - Compares lengths of preprocessed GT and output; caps at 1.
  - Good vs Bad: 1.0 ideal; <1 indicates missing text.
- Mathematical / Algorithmic:
  - completeness = min(1.0, |hyp| / max(1, |gt|)).
- Code Implementation (From Framework):
  - See `compute_all_metrics()` in [ocr_eval_utils.py](ocr_eval_utils.py#L192-L216).
- Example Calculation:
  - GT length=1000 chars; Hyp length=800 → 0.8.
  - Final Score: 0.8
- Strengths:
  - Simple coverage measure; catches truncation.
- Limitations:
  - Ignores correctness; can be 1.0 with hallucinations.
- Common Pitfalls:
  - Treating completeness as accuracy; not aligning with document type.
- Performance Considerations:
  - Trivial to compute; real-time friendly.
- Tuning & Configuration:
  - Apply the same preprocessing used for accuracy metrics.
- Related Metrics:
  - WER/CER; structural metrics.
- Practical Recommendation:
  - Use alongside accuracy metrics to avoid false positives from verbose outputs.

---

## Heading Alignment

- What This Metric Measures:
  - Aspect: Preservation and alignment of markdown headings (titles/subtitles) between GT and output.
  - Importance: Structural fidelity of documents; impacts usability.
- When to Use This Metric:
  - Best: Document OCR/LLM extraction preserving headings.
  - Not reliable: Non-markdown outputs; heavily paraphrased titles.
- How It Works (Conceptual):
  - Greedy ordered matching of normalized headings using fuzzy similarity; multiplies average similarity by coverage.
  - Good vs Bad: 1.0 indicates well-matched headings with correct order; 0.0 no matches.
- Mathematical / Algorithmic:
  - Steps: extract headings; normalize; greedy match with threshold; compute average similarity and coverage (matched/expected); score = avg_sim × coverage; clamp [0,1].
- Code Implementation (From Framework):
  - See `analyze_markdown_structure()` in [ocr_eval_utils.py](ocr_eval_utils.py#L100-L168).
- Example Calculation:
  - GT headings: [“Introduction”, “Methods”]; Hyp: [“Intro”, “Methodology”]; both matched → high similarity and coverage → score ~0.8–1.0.
  - Final Score: e.g., 0.85
- Strengths:
  - Captures both presence and quality of alignment; order-aware.
- Limitations:
  - Fuzzy thresholds heuristic; sensitive to normalization.
- Common Pitfalls:
  - Non-Markdown headings missed; overly aggressive normalization.
- Performance Considerations:
  - Linear-time matching over heading lists; fast.
- Tuning & Configuration:
  - Threshold `THRESH=0.80` baked into implementation; can be adjusted if needed.
- Related Metrics:
  - Section Ordering; Table Preservation.
- Practical Recommendation:
  - Monitor with Section Ordering to detect reordering issues.

---

## List Accuracy

- What This Metric Measures:
  - Aspect: Preservation of list items (unordered/ordered) count.
  - Importance: Structural correctness for content with lists.
- When to Use This Metric:
  - Best: Documents containing lists; check enumerations.
  - Not reliable: Content where list formatting is irrelevant.
- How It Works (Conceptual):
  - Compares counts of list markers; proportional difference.
  - Good vs Bad: 1.0 perfect; 0.0 severe mismatch.
- Mathematical / Algorithmic:
  - Score = 1 − |gt_lists − hyp_lists| / max(1, max(gt_lists, hyp_lists)); clamp [0,1].
- Code Implementation (From Framework):
  - See `analyze_markdown_structure()` in [ocr_eval_utils.py](ocr_eval_utils.py#L169-L216).
- Example Calculation:
  - GT has 10 list bullets; Hyp has 8 → 1 − |10−8|/10 = 0.8.
  - Final Score: 0.8
- Strengths:
  - Simple preservation measure.
- Limitations:
  - Ignores list content fidelity; only counts structure.
- Common Pitfalls:
  - Treating it as semantic correctness; forgetting ordered vs unordered nuances.
- Performance Considerations:
  - Fast.
- Tuning & Configuration:
  - Regex patterns for bullets and ordered lists.
- Related Metrics:
  - Heading Alignment; Section Ordering.
- Practical Recommendation:
  - Combine with text metrics to ensure content quality inside lists.

---

## Table Preservation

- What This Metric Measures:
  - Aspect: Preservation of markdown table presence, column structure, and cell content.
  - Importance: Critical for tabular data integrity.
- When to Use This Metric:
  - Best: Documents with tables.
  - Not reliable: Non-table formats; extremely noisy OCR where table detection is unreliable.
- How It Works (Conceptual):
  - Combines presence score, column-structure similarity, and content Jaccard similarity with weights.
  - Good vs Bad: Higher is better; `None` when no tables present in both.
- Mathematical / Algorithmic:
  - presence = min(gt_tables, hyp_tables) / max(1, max(gt_tables, hyp_tables))
  - col_score = mean over estimated columns per line: min(a,b)/max(1,max(a,b))
  - content_score = |intersection| / max(1, |union|)
  - table_preservation = 0.4 × presence + 0.3 × col_score + 0.3 × content_score; clamp [0,1].
- Code Implementation (From Framework):
  - See `analyze_markdown_structure()` in [ocr_eval_utils.py](ocr_eval_utils.py#L187-L262).
- Example Calculation:
  - GT has 5 table lines, Hyp has 4; columns largely match; content overlap moderate → score ~0.6–0.8.
  - Final Score: e.g., 0.72
- Strengths:
  - Balanced view of table fidelity.
- Limitations:
  - Heuristic; depends on markdown formatting consistency.
- Common Pitfalls:
  - Over-trusting column estimation; ignoring merged cells.
- Performance Considerations:
  - Moderate but fast; linear in number of table lines.
- Tuning & Configuration:
  - Weights (0.4/0.3/0.3) can be tuned if needed.
- Related Metrics:
  - Heading Alignment; Link Correctness.
- Practical Recommendation:
  - Track with List Accuracy to ensure holistic structural preservation.

---

## Link Correctness

- What This Metric Measures:
  - Aspect: Preservation of link targets in markdown.
  - Importance: Functional correctness for linked documents.
- When to Use This Metric:
  - Best: Documents containing links; technical docs.
  - Not reliable: When links are transformed or shortened differently.
- How It Works (Conceptual):
  - Jaccard similarity of normalized link targets.
  - Good vs Bad: 1.0 identical sets; 0.0 no overlap.
- Mathematical / Algorithmic:
  - Score = |intersection| / max(1, |union|).
- Code Implementation (From Framework):
  - See `analyze_markdown_structure()` in [ocr_eval_utils.py](ocr_eval_utils.py#L216-L241).
- Example Calculation:
  - GT links: {a,b,c}; Hyp: {a,c,d} → inter={a,c}=2, union={a,b,c,d}=4 → 0.5.
  - Final Score: 0.5
- Strengths:
  - Simple, robust to order.
- Limitations:
  - Ignores anchor texts; only targets.
- Common Pitfalls:
  - Normalization issues (case, protocol, tracking params).
- Performance Considerations:
  - Fast set operations.
- Tuning & Configuration:
  - Normalization function `_norm()` affects robustness.
- Related Metrics:
  - Section Ordering; Heading Alignment.
- Practical Recommendation:
  - Pair with WER/CER to ensure linked content also matches.

---

## Section Ordering

- What This Metric Measures:
  - Aspect: Preservation of section ordering implied by matched headings.
  - Importance: Usability and navigability of extracted documents.
- When to Use This Metric:
  - Best: Structured documents with clear section flow.
  - Not reliable: Flat documents without headings; heavily reorganized outputs.
- How It Works (Conceptual):
  - Computes order consistency by approximating LCS of matched heading indices; scales by coverage.
  - Good vs Bad: 1.0 perfect order; lower values indicate reordering.
- Mathematical / Algorithmic:
  - Steps: Compute matched pairs; derive index sequence; approximate LIS/LCS length; score = (LCS / matches) × coverage; clamp [0,1].
- Code Implementation (From Framework):
  - See `analyze_markdown_structure()` in [ocr_eval_utils.py](ocr_eval_utils.py#L140-L186).
- Example Calculation:
  - Matched heading indices strictly increasing → LCS equals matches → high score.
  - Final Score: e.g., 0.9
- Strengths:
  - Captures ordering fidelity beyond heading similarity.
- Limitations:
  - Approximation (LIS used); sensitive to matching quality.
- Common Pitfalls:
  - Sparse matches yield unstable ordering; threshold too strict.
- Performance Considerations:
  - Fast for typical heading counts.
- Tuning & Configuration:
  - Matching threshold affects ordering; may be tuned.
- Related Metrics:
  - Heading Alignment.
- Practical Recommendation:
  - Monitor together with Heading Alignment for a complete structural view.

---

## Aggregated Mean (Multi-run)

- What This Metric Measures:
  - Aspect: Average value per metric across multiple runs.
  - Importance: Summarizes central tendency; reduces noise.
- When to Use This Metric:
  - Best: Batch evaluations; stability analysis.
  - Not reliable: With heavy outliers and very few runs.
- How It Works (Conceptual):
  - Arithmetic mean of numeric metric values across runs (ignoring `None`).
  - Good vs Bad: Depends on metric—lower is better for error rates, higher for structural scores.
- Mathematical / Algorithmic:
  - mean = average(valid values).
- Code Implementation (From Framework):
  - See `compute_aggregated_metric()` in [multi_run_evaluation.py](multi_run_evaluation.py#L81-L135).
- Example Calculation:
  - Runs WER: [0.2, 0.25, 0.3] → mean=0.25.
- Strengths:
  - Simple, robust to small variance.
- Limitations:
  - Sensitive to outliers.
- Common Pitfalls:
  - Interpreting mean without variance context.
- Performance Considerations:
  - Trivial compute.
- Tuning & Configuration:
  - Optional exclusion of `None` values.
- Related Metrics:
  - Std Dev, CCI.
- Practical Recommendation:
  - Always present mean with Std Dev and CCI.

---

## Standard Deviation (Multi-run)

- What This Metric Measures:
  - Aspect: Spread/variance across runs.
  - Importance: Indicates stability of metric across stochastic runs.
- When to Use This Metric:
  - Best: Any metric with variable outputs per run.
  - Not reliable: Very few runs (n<2 yields 0).
- How It Works (Conceptual):
  - Sample standard deviation over valid values.
  - Good vs Bad: Lower is better for stability.
- Mathematical / Algorithmic:
  - std_dev = sample standard deviation (`statistics.stdev`) when n>1 else 0.0.
- Code Implementation (From Framework):
  - See `compute_aggregated_metric()` in [multi_run_evaluation.py](multi_run_evaluation.py#L120-L170).
- Example Calculation:
  - [0.2, 0.25, 0.3] → std≈0.05.
- Strengths:
  - Quantifies variability.
- Limitations:
  - Needs enough runs; affected by outliers.
- Common Pitfalls:
  - Interpreting std without mean context.
- Performance Considerations:
  - Fast.
- Tuning & Configuration:
  - None.
- Related Metrics:
  - CCI.
- Practical Recommendation:
  - Use CCI to contextualize std relative to mean.

---

## Consistency Confidence Index (CCI)

- What This Metric Measures:
  - Aspect: Stability of a metric across runs relative to its mean.
  - Importance: Helps decide reliability of reported metric values.
- When to Use This Metric:
  - Best: Multi-run evaluations; comparing models’ consistency.
  - Not reliable: When mean is ~0 (division protection yields 0).
- How It Works (Conceptual):
  - CCI = 1 − (std_dev / mean) if mean>0 else 0; higher is more stable.
  - Good vs Bad: >0.90 highly stable; ≥0.75 moderately stable; else unstable.
- Mathematical / Algorithmic:
  - CCI = max(0, 1 − (σ/μ)).
- Code Implementation (From Framework):
  - See `compute_aggregated_metric()` in [multi_run_evaluation.py](multi_run_evaluation.py#L136-L170).
- Example Calculation:
  - mean=0.25, std=0.05 → CCI=1−0.05/0.25=0.8.
- Strengths:
  - Simple interpretability; relative measure.
- Limitations:
  - Undefined when μ≈0; aggressive penalty when mean small.
- Common Pitfalls:
  - Comparing CCI across metrics with different scales.
- Performance Considerations:
  - Trivial compute.
- Tuning & Configuration:
  - Interpretation bands in reporter.
- Related Metrics:
  - Std Dev; Variance Trend.
- Practical Recommendation:
  - Use CCI alongside mean/std to judge reliability.

---

## Variance Trend (Multi-run)

- What This Metric Measures:
  - Aspect: Directionality of metric changes across runs.
  - Importance: Detects drift/improvement/degradation.
- When to Use This Metric:
  - Best: ≥3 runs; monitoring over repeated evaluations.
  - Not reliable: Fewer than 3 runs.
- How It Works (Conceptual):
  - Compare mean of first half vs second half; categorize as stable/increasing/decreasing; suffix `_with_na` if missing values present.
- Mathematical / Algorithmic:
  - diff_ratio = |μ2−μ1|/max(μ1, 0.001); if <5% → stable; else increasing/decreasing by comparison.
- Code Implementation (From Framework):
  - See `compute_aggregated_metric()` in [multi_run_evaluation.py](multi_run_evaluation.py#L149-L170).
- Example Calculation:
  - [0.30, 0.28, 0.25] → decreasing.
- Strengths:
  - Quick directional insight.
- Limitations:
  - Heuristic; coarse.
- Common Pitfalls:
  - Overinterpreting small shifts; ignoring run order effects.
- Performance Considerations:
  - Trivial compute.
- Tuning & Configuration:
  - Threshold (5%) can be adjusted if needed.
- Related Metrics:
  - CCI.
- Practical Recommendation:
  - Use with CCI to assess both stability and trajectory across runs.

---

## Composite Scores (Multi-run)

### Text Accuracy Score

- What This Metric Measures:
  - Aspect: Aggregate lexical accuracy from multiple error metrics.
  - Importance: Single score for text correctness.
- How It Works:
  - `Text Score = 1 − average(WER, CER, MER, WIL)`; reporter may include `lev_norm` in average.
- Code Implementation:
  - See `compute_composite_scores()` in [multi_run_evaluation.py](multi_run_evaluation.py#L171-L233). Reporter backfills with `lev_norm` in [multi_run_reporter.py](multi_run_reporter.py#L116-L171).
- Example:
  - WER=0.2, CER=0.1, MER=0.15, WIL=0.18 → avg_err=0.1575 → score=0.8425.

### Structural Score

- What This Metric Measures:
  - Aspect: Aggregate structural fidelity.
- How It Works:
  - Average of [Heading Alignment, List Accuracy, Table Preservation (ignore NA), Link Correctness, Section Ordering].
- Code Implementation:
  - See `compute_composite_scores()` in [multi_run_evaluation.py](multi_run_evaluation.py#L203-L233).
- Example:
  - [0.9, 0.8, 0.7, 0.9, 0.85] → score=0.83.

### Overall Extraction Score

- What This Metric Measures:
  - Aspect: Combined text + structural quality.
- How It Works:
  - `Overall = 0.6 × Text Score + 0.4 × Structural Score`.
- Code Implementation:
  - See `compute_composite_scores()` in [multi_run_evaluation.py](multi_run_evaluation.py#L229-L233).
- Example:
  - Text=0.8425, Structural=0.83 → Overall=0.6×0.8425+0.4×0.83≈0.838.

---

## Diagnostic: Top Word Mismatches

- Purpose: Identify most frequent word-level replacements/insertions/deletions to guide fixes.
- Implementation: `compute_top_word_mismatches()` in [ocr_eval_utils.py](ocr_eval_utils.py#L263-L303).
- Note: This is diagnostic, not an evaluation score.

---

## How Metrics Work Together

- Lexical Similarity:
  - WER, MER, WIL, CER, Levenshtein, `lev_norm`.
- Semantic Similarity:
  - Not directly measured; consider pairing with semantic metrics (external) if needed.
- Factual Correctness:
  - Not directly measured; future work may include grounding checks.
- Structure/Format Adherence:
  - Heading Alignment, List Accuracy, Table Preservation, Link Correctness, Section Ordering.

- Practical Guidance:
  - Use Text Accuracy Score to summarize lexical performance; include `lev_norm` for robustness.
  - Use Structural Score to ensure format fidelity.
  - Use Overall Extraction Score for product-facing dashboards.
  - Use CCI and Variance Trend to judge reliability and trajectory across runs.

---

## References

- JiWER library: [Documentation](https://jitsi.github.io/jiwer/) and [Source code](https://github.com/jitsi/jiwer).
- RapidFuzz: [Repository](https://github.com/maxbachmann/RapidFuzz).
