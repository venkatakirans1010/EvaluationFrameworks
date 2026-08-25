# Expected Evaluation Results Snapshot

## 📊 Evaluation Metrics Display

When you run an evaluation with reference text, you should see the following structure:

### ✅ Model Response Card

```
┌─────────────────────────────────────────┐
│ ✅ gemini-2.5-flash                     │
│ Provider: RouteLLM                      │
│                                         │
│ ┌─────────────────────────────────────┐ │
│ │ Response                            │ │
│ │ [Model's generated response text]   │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ Tokens: 1,234 (Prompt: 100, Completion: 1,134) │
│                                         │
│ ┌─────────────────────────────────────┐ │
│ │ 📊 Evaluation Metrics (BLEU/ROUGE)  │ │
│ │                                     │ │
│ │ #### BLEU Scores                    │ │
│ │ ┌──────┬───────┬───────┬──────┬────┐│
│ │ │ BLEU │ BLEU-1│ BLEU-2│BLEU-3│BLEU││
│ │ │      │       │       │      │ -4 ││
│ │ │0.1234│0.5678 │0.2345 │0.1234│0.05││
│ │ └──────┴───────┴───────┴──────┴────┘│
│ │                                     │ │
│ │ #### ROUGE Scores                   │ │
│ │                                     │ │
│ │ **ROUGE-1:**                        │ │
│ │ ┌───────────┬────────┬──────────┐  │ │
│ │ │ Precision │ Recall │    F1    │  │ │
│ │ │   0.6789  │ 0.5432 │  0.6123  │  │ │
│ │ └───────────┴────────┴──────────┘  │ │
│ │                                     │ │
│ │ **ROUGE-2:**                        │ │
│ │ ┌───────────┬────────┬──────────┐  │ │
│ │ │ Precision │ Recall │    F1    │  │ │
│ │ │   0.4567  │ 0.3210 │  0.3789  │  │ │
│ │ └───────────┴────────┴──────────┘  │ │
│ │                                     │ │
│ │ **ROUGE-L:**                        │ │
│ │ ┌───────────┬────────┬──────────┐  │ │
│ │ │ Precision │ Recall │    F1    │  │ │
│ │ │   0.6543  │ 0.5123 │  0.5789  │  │ │
│ │ └───────────┴────────┴──────────┘  │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

## 📈 Expected Score Ranges

### BLEU Scores
- **BLEU**: 0.0 to 1.0 (higher is better)
  - 0.0-0.3: Poor match
  - 0.3-0.6: Moderate match
  - 0.6-0.8: Good match
  - 0.8-1.0: Excellent match

- **BLEU-1**: Unigram precision (word-level matching)
- **BLEU-2**: Bigram precision (2-word phrase matching)
- **BLEU-3**: Trigram precision (3-word phrase matching)
- **BLEU-4**: 4-gram precision (4-word phrase matching)

### ROUGE Scores
- **ROUGE-1**: Unigram-based (word overlap)
  - Precision: How many reference words appear in candidate
  - Recall: How many candidate words appear in reference
  - F1: Harmonic mean of precision and recall

- **ROUGE-2**: Bigram-based (2-word phrase overlap)
  - Measures overlap of word pairs

- **ROUGE-L**: Longest Common Subsequence
  - Measures longest matching sequence of words
  - Better for measuring sentence-level similarity

## 🔍 Example Values

For a well-matched response:
```
BLEU: 0.4523
BLEU-1: 0.6789
BLEU-2: 0.4567
BLEU-3: 0.2345
BLEU-4: 0.1234

ROUGE-1:
  Precision: 0.7890
  Recall: 0.6543
  F1: 0.7123

ROUGE-2:
  Precision: 0.5678
  Recall: 0.4321
  F1: 0.4890

ROUGE-L:
  Precision: 0.7654
  Recall: 0.6210
  F1: 0.6876
```

## ⚠️ Troubleshooting

If ROUGE scores are not showing:
1. Check that reference text was provided (PDF upload or text input)
2. Verify both reference and candidate text are not empty
3. Check browser console for any JavaScript errors
4. Look for error messages in the metrics section

## 📝 Notes

- BLEU scores are typically lower than ROUGE scores
- ROUGE-1 F1 is usually the highest among ROUGE metrics
- Scores depend on the similarity between reference and generated text
- Very different texts will have low scores across all metrics

