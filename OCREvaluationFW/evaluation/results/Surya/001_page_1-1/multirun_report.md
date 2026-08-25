# 📊 Multi-Run Evaluation Report

**📄 Document:** `001_page_1-1`
**🤖 Model:** `Surya`
**🕒 Generated:** 2025-12-24 18:34:20
**📈 Total Runs:** 3
**✅ Successful Runs:** 0
**❌ Failed Runs:** 3

## 🎯 Executive Summary

**Overall Consistency Confidence Index (CCI):** `0.0000`
**Stability Interpretation:** Insufficient data

🔴 **Poor Consistency:** The model shows significant variance between runs, indicating potential reliability issues.

## 🧮 Aggregate Scores

| Metric | Score |
|--------|-------|
| **Text Accuracy Score** | NA |
| **Structural Score** | NA |
| **Overall Extraction Score** | NA |

- Text Accuracy Score = 1 - average(WER, CER, MER, WIL, LEV-NORM)
- Structural Score = average(Heading Alignment, List Accuracy, Table Preservation, Link Correctness, Section Ordering) — Table Preservation ignored if NA
- Overall Extraction Score = 0.6 × Text Score + 0.4 × Structural Score

## 🔄 Run Execution Details

| Run | Status | Duration (s) | Error |
|-----|--------|--------------|-------|
| **1** | ❌ Failed | 7.65 | All configured Datalab endpoints failed. Tried:... |
| **2** | ❌ Failed | 8.10 | All configured Datalab endpoints failed. Tried:... |
| **3** | ❌ Failed | 7.88 | All configured Datalab endpoints failed. Tried:... |

## 💡 Recommendations

❌ **Model Performance:** Poor consistency. Investigate model configuration and parameters.
❌ **Deployment:** Not recommended for production without addressing consistency issues.

## 🔧 Technical Details

**Consistency Confidence Index (CCI) Formula:** `CCI = 1 - (StdDev / Mean)`

**Interpretation Guidelines:**
- CCI > 0.90: Highly stable
- CCI 0.75-0.90: Moderately stable
- CCI < 0.75: Unstable results

**Metrics Explanation:**
- **WER**: Word Error Rate (lower is better)
- **MER**: Match Error Rate (lower is better)
- **WIL**: Word Information Lost (lower is better)
- **CER**: Character Error Rate (lower is better)
- **LEV-NORM**: Normalized Levenshtein Distance (lower is better)
- **Completeness**: Ratio of OCR length to ground truth length (higher is better)

**Structural Analysis Metrics:**
- **Heading Alignment**: Accuracy of heading structure preservation (higher is better)
- **List Accuracy**: Accuracy of list structure preservation (higher is better)
- **Table Preservation**: Accuracy of table structure preservation (higher is better)
- **Link Correctness**: Accuracy of link preservation (higher is better)
- **Section Ordering**: Accuracy of section order preservation (higher is better)