# 📊 OCR Evaluation Report

**🤖 Model:** `Qwen3 VL 8B – via OpenRouter`  
**📄 Ground Truth:** `class-10-sample-paper-2020-21-telugu-pages-1.md (Ground Truth)`  
**📝 OCR Output:** `class-10-sample-paper-2020-21-telugu-pages-1.pdf (OCR Output)`  
**🕒 Timestamp:** 2026-01-22 16:23:11  

## 📊 Unified Metrics

| WER | MER | WIL | CER | LEV-DIST (chars) | Completeness | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |
|-----|-----|-----|-----|-------------------|--------------|-------------------|---------------|--------------------|------------------|------------------|
| 1.0000 | 1.0000 | 1.0000 | 0.9959 | 2184 | 0.0078 | 1.0000 | 1.0000 | NA | 1.0000 | 1.0000 |


*Lower values indicate better OCR performance (except Completeness where 1.0 is ideal). Structural metrics range from 0.0–1.0 (higher is better).*

## 🔍 Top Word Mismatches

| Ground Truth | OCR Output | Count |
|---|---|---:|
| `TELUGU` | `\n)` | **1** |
| `(CODE:` | `)` | **1** |
| `007)` | `}` | **1** |
| `Class` | `。` | **1** |
| `-` | `;` | **1** |
| `X` | `}` | **1** |
| `(2020` | `。` | **1** |
| `-` | `}` | **1** |