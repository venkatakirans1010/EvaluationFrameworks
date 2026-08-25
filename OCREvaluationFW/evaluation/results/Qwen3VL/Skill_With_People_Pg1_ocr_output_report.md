# 📊 OCR Evaluation Report

**🤖 Model:** `Qwen3 VL 8B – via OpenRouter`  
**📄 Ground Truth:** `Skill_With_People_Pg1.md (Ground Truth)`  
**📝 OCR Output:** `Skill_With_People_Pg1.pdf (OCR Output)`  
**🕒 Timestamp:** 2026-01-21 12:18:16  

## 📊 Unified Metrics

| WER | MER | WIL | CER | LEV-DIST (chars) | Completeness | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |
|-----|-----|-----|-----|-------------------|--------------|-------------------|---------------|--------------------|------------------|------------------|
| 0.0292 | 0.0292 | 0.0521 | 0.0131 | 12 | 0.9956 | 0.0000 | 1.0000 | NA | 1.0000 | 1.0000 |


*Lower values indicate better OCR performance (except Completeness where 1.0 is ideal). Structural metrics range from 0.0–1.0 (higher is better).*

## 🔍 Top Word Mismatches

| Ground Truth | OCR Output | Count |
|---|---|---:|
| `"yes"` | `“yes”` | **4** |
| `###` | `<MISSING>` | **1** |