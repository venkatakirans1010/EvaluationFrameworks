# 📊 OCR Evaluation Report

**🤖 Model:** `Qwen3 VL 8B – via OpenRouter`  
**📄 Ground Truth:** `Skill_With_People_Pg3.md`  
**📝 OCR Output:** `Skill_With_People_Pg3.pdf`  
**🕒 Timestamp:** 2026-01-22 11:57:26  

## 📊 Unified Metrics

| WER | MER | WIL | CER | LEV-DIST (chars) | Completeness | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |
|-----|-----|-----|-----|-------------------|--------------|-------------------|---------------|--------------------|------------------|------------------|
| 0.0983 | 0.0958 | 0.1615 | 0.0208 | 26 | 1.0000 | 1.0000 | 1.0000 | NA | 1.0000 | 1.0000 |


*Lower values indicate better OCR performance (except Completeness where 1.0 is ideal). Structural metrics range from 0.0–1.0 (higher is better).*

## 🔍 Top Word Mismatches

| Ground Truth | OCR Output | Count |
|---|---|---:|
| `"Do` | `“Do` | **5** |
| `won't` | `won’t` | **2** |
| `"Would` | `“Would` | **1** |
| `afternoon?"` | `afternoon?”` | **1** |
| `can't` | `can’t` | **1** |
| `-` | `–` | **1** |
| `white?"` | `white?”` | **1** |
| `these?")` | `these?”)` | **1** |
| `"So` | `“So` | **1** |
| `Tuesday?"` | `Tuesday?”` | **1** |