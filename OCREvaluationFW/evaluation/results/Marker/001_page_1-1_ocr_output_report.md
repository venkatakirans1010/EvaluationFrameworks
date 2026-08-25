# 📊 OCR Evaluation Report

**🤖 Model:** `Marker – Convert to Markdown/HTML/JSON`  
**📄 Ground Truth:** `001_page_1-1.md (Ground Truth)`  
**📝 OCR Output:** `001_page_1-1.pdf (OCR Output)`  
**🕒 Timestamp:** 2026-01-08 13:15:14  

## 📊 Unified Metrics

| WER | MER | WIL | CER | LEV-DIST (chars) | Completeness | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |
|-----|-----|-----|-----|-------------------|--------------|-------------------|---------------|--------------------|------------------|------------------|
| 0.1434 | 0.1418 | 0.2273 | 0.1067 | 182 | 1.0000 | 1.0000 | 0.0000 | NA | 1.0000 | 1.0000 |


*Lower values indicate better OCR performance (except Completeness where 1.0 is ideal). Structural metrics range from 0.0–1.0 (higher is better).*

## 🔍 Top Word Mismatches

| Ground Truth | OCR Output | Count |
|---|---|---:|
| `●` | `-` | **8** |
| `<EXTRA>` | `{0}------------------------------------------------` | **1** |
| `TL;DR:` | `**TL;DR:**` | **1** |
| `deﬁned` | `defined` | **1** |
| `evaluate` | `[evaluate` | **1** |
| `quality.` | `quality](#).` | **1** |
| `you’re` | `you're` | **1** |
| `workﬂows.` | `workflows.` | **1** |
| `deﬁne.` | `define.` | **1** |
| `Politeness:` | `**Politeness:**` | **1** |