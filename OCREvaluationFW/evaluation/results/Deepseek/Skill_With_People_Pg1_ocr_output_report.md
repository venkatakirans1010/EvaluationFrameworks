# 📊 OCR Evaluation Report

**🤖 Model:** `Deepseek – OCR via DeepInfra`  
**📄 Ground Truth:** `Skill_With_People_Pg1.md (Ground Truth)`  
**📝 OCR Output:** `Skill_With_People_Pg1.pdf (OCR Output)`  
**🕒 Timestamp:** 2026-01-08 11:55:20  

## 📊 Unified Metrics

| WER | MER | WIL | CER | LEV-NORM | Completeness | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |
|-----|-----|-----|-----|----------|--------------|-------------------|---------------|--------------------|------------------|------------------|
| 0.0585 | 0.0585 | 0.1135 | 1.0305 | 0.0196 | 0.9967 | 0.0000 | 0.2000 | NA | 1.0000 | 1.0000 |


*Lower values indicate better OCR performance (except Completeness where 1.0 is ideal). Structural metrics range from 0.0–1.0 (higher is better).*

## 🔍 Top Word Mismatches

| Ground Truth | OCR Output | Count |
|---|---|---:|
| `"yes"` | `“yes”` | **4** |
| `•` | `-` | **4** |
| `###` | `8` | **1** |
| `8` | `<MISSING>` | **1** |