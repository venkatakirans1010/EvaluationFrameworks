# 📊 OCR Evaluation Report

**🤖 Model:** `Marker – Convert to Markdown/HTML/JSON`  
**📄 Ground Truth:** `Skill_With_People_Pg1.md (Ground Truth)`  
**📝 OCR Output:** `Skill_With_People_Pg1.pdf (OCR Output)`  
**🕒 Timestamp:** 2026-01-08 13:19:14  

## 📊 Unified Metrics

| WER | MER | WIL | CER | LEV-DIST (chars) | Completeness | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |
|-----|-----|-----|-----|-------------------|--------------|-------------------|---------------|--------------------|------------------|------------------|
| 0.1287 | 0.1264 | 0.2052 | 0.0892 | 82 | 1.0000 | 0.5000 | 0.2000 | NA | 1.0000 | 0.5000 |


*Lower values indicate better OCR performance (except Completeness where 1.0 is ideal). Structural metrics range from 0.0–1.0 (higher is better).*

## 🔍 Top Word Mismatches

| Ground Truth | OCR Output | Count |
|---|---|---:|
| `•` | `-` | **5** |
| `"yes"` | `“yes”` | **4** |
| `###` | `{0}------------------------------------------------` | **1** |
| `Here` | `**Here` | **1** |
| `–` | `–**` | **1** |
| `1.` | `##` | **1** |
| `you.` | `you.**` | **1** |