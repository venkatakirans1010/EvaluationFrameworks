# 📊 OCR Evaluation Report

**🤖 Model:** `Marker – Convert to Markdown/HTML/JSON`  
**📄 Ground Truth:** `CurrentAdangal.md`  
**📝 OCR Output:** `CurrentAdangal.pdf`  
**🕒 Timestamp:** 2026-01-08 19:30:34  

## 📊 Unified Metrics

| WER | MER | WIL | CER | LEV-DIST (chars) | Completeness | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |
|-----|-----|-----|-----|-------------------|--------------|-------------------|---------------|--------------------|------------------|------------------|
| 0.3377 | 0.2950 | 0.3939 | 0.4158 | 474 | 1.0000 | 0.5000 | 0.5000 | 0.8573 | 1.0000 | 0.5000 |


*Lower values indicate better OCR performance (except Completeness where 1.0 is ideal). Structural metrics range from 0.0–1.0 (higher is better).*

## 🔍 Top Word Mismatches

| Ground Truth | OCR Output | Count |
|---|---|---:|
| `<EXTRA>` | `|` | **14** |
| `|` | `<MISSING>` | **10** |
| `:` | `|` | **8** |
| `:---` | `<MISSING>` | **4** |
| `|` | `|` | **4** |
| `ఫసలీ)` | `ఫసరీ)` | **2** |
| `|` | `:` | **2** |
| `1.` | `|` | **2** |
| `మీ` | `{0}------------------------------------------------` | **1** |
| `సేవ` | `![](935eed7aa61f7777f62cfc032e11bee9_img.jpg)` | **1** |