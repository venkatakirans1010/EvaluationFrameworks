# 📊 OCR Evaluation Report

**🤖 Model:** `Marker – Convert to Markdown/HTML/JSON`  
**📄 Ground Truth:** `Images_And_Charts.md (Ground Truth)`  
**📝 OCR Output:** `Images_And_Charts.pdf (OCR Output)`  
**🕒 Timestamp:** 2025-12-22 18:27:42  

## 📊 Unified Metrics

| WER | MER | WIL | CER | LEV-NORM | Completeness | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |
|-----|-----|-----|-----|----------|--------------|-------------------|---------------|--------------------|------------------|------------------|
| 0.4508 | 0.4394 | 0.6288 | 1.3866 | 0.5110 | 1.0000 | 0.0000 | 1.0000 | 0.9032 | 1.0000 | 1.0000 |


*Lower values indicate better OCR performance (except Completeness where 1.0 is ideal). Structural metrics range from 0.0–1.0 (higher is better).*

## 🔍 Top Word Mismatches

| Ground Truth | OCR Output | Count |
|---|---|---:|
| `with` | `showing` | **2** |
| `series,` | `series` | **2** |
| `(light` | `across` | **2** |
| `items.` | `items` | **2** |
| `The` | `(Item` | **2** |
| `y-axis` | `1,` | **2** |
| `ranges` | `Item` | **2** |
| `from` | `2,` | **2** |
| `0` | `Item` | **2** |
| `<EXTRA>` | `{0}------------------------------------------------` | **1** |