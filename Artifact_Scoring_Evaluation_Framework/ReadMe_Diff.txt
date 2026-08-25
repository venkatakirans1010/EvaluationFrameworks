# ReadMe Diff: Newly Added Metrics (Temporary Notes)

## Baseline (Already Existing)
The framework originally scored artifacts using 6 heuristic dimensions (1–5 scale):
1. Faithfulness
2. Coverage
3. Relevance
4. Conciseness
5. Clarity
6. Structure

These are strong for general quality screening, but they are mostly heuristic and do not explicitly capture ranking quality, semantic similarity depth, redundancy, or label-based classification/entity performance.

---

## Newly Added Metrics (What changed)

### 1) Summary Evaluation

#### ROUGE-1
- Brief summary: Unigram lexical overlap between generated summary and reference context.
- Key advantage: Adds direct, interpretable recall-style overlap signal.
- How it differs from existing metrics:
  - Existing Coverage/Relevance are heuristic overlap dimensions on 1–5 scale.
  - ROUGE-1 is a normalized lexical overlap score (0–1) and is more standard for summarization benchmarking.

#### ROUGE-L
- Brief summary: Longest Common Subsequence based overlap between summary and reference.
- Key advantage: Captures sequence-level alignment, not just bag-of-words overlap.
- How it differs from existing metrics:
  - Existing metrics do not explicitly model sequence continuity.
  - ROUGE-L rewards preserving structure/order from source context.

#### BERTScore (optional)
- Brief summary: Semantic similarity score using contextual embeddings.
- Key advantage: Detects semantic equivalence even when wording differs.
- How it differs from existing metrics:
  - Existing six metrics are primarily lexical/heuristic.
  - BERTScore captures meaning-level similarity beyond exact token overlap.

#### Compression Ratio
- Brief summary: summary length divided by context length.
- Key advantage: Quantifies compression directly (how concise output really is).
- How it differs from existing metrics:
  - Existing Conciseness is a heuristic target-based score.
  - Compression Ratio is raw/explicit and easier to trend over time.

#### TF-IDF Overlap (Summary)
- Brief summary: Weighted lexical similarity emphasizing informative terms.
- Key advantage: Reduces impact of frequent/common words and stresses meaningful terms.
- How it differs from existing metrics:
  - Existing Coverage treats tokens more uniformly.
  - TF-IDF overlap better reflects content-bearing term alignment.

---

### 2) FAQ Evaluation

#### Question Relevance
- Brief summary: Similarity of FAQ questions to reference context.
- Key advantage: Detects off-topic questions early.
- How it differs from existing metrics:
  - Existing Relevance is generic at artifact level.
  - Question Relevance specifically isolates the question side of FAQ quality.

#### Answer Correctness
- Brief summary: Similarity/alignment of answers with reference context.
- Key advantage: Separates answer grounding from question quality.
- How it differs from existing metrics:
  - Existing Faithfulness applies broadly.
  - Answer Correctness is FAQ-answer-specific, giving finer diagnostics.

#### Redundancy Score
- Brief summary: Pairwise similarity among FAQ items.
- Key advantage: Highlights repetition and duplicate content.
- How it differs from existing metrics:
  - Existing framework did not directly measure cross-item duplication.
  - Redundancy is a set-level diversity metric, not a per-item quality score.

---

### 3) Keyword Evaluation

#### Precision@K
- Brief summary: Fraction of top-K predicted keywords that match reference context.
- Key advantage: Measures top-ranked keyword quality directly.
- How it differs from existing metrics:
  - Existing Keyword scoring used generic 6-dimension heuristics.
  - Precision@K is ranking-aware and standard in retrieval/extraction tasks.

#### Recall@K
- Brief summary: Fraction of important reference terms covered by top-K predicted keywords.
- Key advantage: Measures coverage adequacy of selected keywords.
- How it differs from existing metrics:
  - Existing Coverage is broad and heuristic.
  - Recall@K explicitly tracks missed important terms in ranked output.

#### TF-IDF Overlap (Keywords)
- Brief summary: Weighted similarity between keyword set and reference content.
- Key advantage: Rewards informative keyword alignment rather than surface overlap.
- How it differs from existing metrics:
  - Existing keyword quality dimensions do not weight term informativeness.
  - TF-IDF overlap provides a stronger signal for true topical relevance.

---

### 4) Entity Evaluation

#### NER F1
- Brief summary: Harmonic mean of entity precision and recall against labeled references.
- Key advantage: Balanced quality metric for entity extraction performance.
- How it differs from existing metrics:
  - Existing entity scoring was heuristic and text-similarity driven.
  - NER F1 is a label-based extraction metric used in standard NLP evaluation.

#### Entity Linking Accuracy
- Brief summary: Percent of entities correctly linked to expected target/entity IDs.
- Key advantage: Measures correctness of disambiguation, not just detection.
- How it differs from existing metrics:
  - Existing metrics evaluate mention quality/overlap.
  - Linking accuracy evaluates resolution correctness (entity identity).

---

### 5) Classification Evaluation

#### Accuracy
- Brief summary: Overall fraction of correctly predicted classes.
- Key advantage: Simple high-level correctness indicator.
- How it differs from existing metrics:
  - Existing classification quality used heuristic dimensions.
  - Accuracy is direct label-level correctness.

#### Precision / Recall / F1
- Brief summary:
  - Precision: of predicted positives, how many are correct.
  - Recall: of true positives, how many were found.
  - F1: balance between precision and recall.
- Key advantage: Better diagnostic visibility than accuracy alone, especially for imbalanced classes.
- How it differs from existing metrics:
  - Existing dimensions do not capture class imbalance behavior.
  - PR/F1 are standard supervised classification metrics.

#### Confusion Matrix
- Brief summary: Table of true labels vs predicted labels.
- Key advantage: Shows exact misclassification patterns.
- How it differs from existing metrics:
  - Existing framework gave aggregate quality scores.
  - Confusion matrix gives class-by-class error analysis.

---

## Practical Benefits of the New Metric Set
1. Better task-specific diagnostics (summary vs FAQ vs keyword vs entity vs classification).
2. More standard NLP benchmarking compatibility (ROUGE, PR/F1, confusion matrix).
3. Stronger ranking-aware evaluation for keywords (Precision@K, Recall@K).
4. Ability to separate lexical quality from semantic quality (TF-IDF overlap vs BERTScore).
5. Better operational decisioning (pinpoint whether issues are relevance, redundancy, ranking, labeling, or linking).

---

## Notes on Availability
- Some metrics may appear as N/A when required inputs are missing (e.g., labeled references for NER/classification/linking) or optional dependencies are not installed (e.g., BERTScore package).
- This behavior is intentional to prevent misleading proxy values.
