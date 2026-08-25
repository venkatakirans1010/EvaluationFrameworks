# 📊 OCR Evaluation Report

**🤖 Model:** `Surya – OCR text extraction`  
**📄 Ground Truth:** `class-10-sample-paper-2020-21-telugu-pages-1.md`  
**📝 OCR Output:** `class-10-sample-paper-2020-21-telugu-pages-1.pdf`  
**🕒 Timestamp:** 2026-01-08 16:56:15  

## 📊 Unified Metrics

| WER | MER | WIL | CER | LEV-DIST (chars) | Completeness | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |
|-----|-----|-----|-----|-------------------|--------------|-------------------|---------------|--------------------|------------------|------------------|
| 0.4362 | 0.4190 | 0.6030 | 0.1601 | 351 | 1.0000 | 1.0000 | 0.0000 | NA | 1.0000 | 1.0000 |


*Lower values indicate better OCR performance (except Completeness where 1.0 is ideal). Structural metrics range from 0.0–1.0 (higher is better).*

## 🔍 Top Word Mismatches

| Ground Truth | OCR Output | Count |
|---|---|---:|
| `వీరేశలింగం` | `వీరేశరింగం` | **2** |
| `<EXTRA>` | `ఈ)` | **2** |
| `TELUGU` | `Downloaded` | **1** |
| `Total:` | `<b>Total:` | **1** |
| `Marks` | `Marks</b>` | **1** |
| `విభాగం` | `<b>విభాగం` | **1** |
| `ఎ` | `ఎ</b>` | **1** |
| `5x1=5` | `1.` | **1** |
| `(అ)` | `<math>5` | **1** |
| `<EXTRA>` | `(e)` | **1** |