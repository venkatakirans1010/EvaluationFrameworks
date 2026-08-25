# 📊 OCR Evaluation Report

**🤖 Model:** `Marker – Convert to Markdown/HTML/JSON`  
**📄 Ground Truth:** `class-10-sample-paper-2020-21-telugu-pages-1.md`  
**📝 OCR Output:** `class-10-sample-paper-2020-21-telugu-pages-1.pdf`  
**🕒 Timestamp:** 2026-01-08 16:50:38  

## 📊 Unified Metrics

| WER | MER | WIL | CER | LEV-DIST (chars) | Completeness | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |
|-----|-----|-----|-----|-------------------|--------------|-------------------|---------------|--------------------|------------------|------------------|
| 0.3745 | 0.3421 | 0.4999 | 0.1149 | 252 | 0.9631 | 0.0000 | 1.0000 | NA | 1.0000 | 0.5000 |


*Lower values indicate better OCR performance (except Completeness where 1.0 is ideal). Structural metrics range from 0.0–1.0 (higher is better).*

## 🔍 Top Word Mismatches

| Ground Truth | OCR Output | Count |
|---|---|---:|
| `సంస్కృతి` | `సంస్థతి` | **3** |
| `సేవ` | `నేవ` | **2** |
| `TELUGU` | `{0}------------------------------------------------` | **1** |
| `<EXTRA>` | `#` | **1** |
| `<EXTRA>` | `##` | **1** |
| `<EXTRA>` | `1.` | **1** |
| `ఏదైనా` | `విడ్డెనా` | **1** |
| `(అ)` | `(ఆ)` | **1** |
| `'రాజశేఖర` | `'రాజశిఖర` | **1** |
| `గాంచినదీనవల.` | `గాంచినదినవల.` | **1** |