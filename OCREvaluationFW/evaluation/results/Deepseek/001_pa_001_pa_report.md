# 📊 OCR Evaluation Report

**🤖 Model:** `Deepseek – OCR via DeepInfra`  
**📄 Ground Truth:** `001_page_1-1.md`  
**📝 OCR Output:** `001_page_1-1.pdf`  
**🕒 Timestamp:** 2026-01-05 15:02:33  

## 📊 Unified Metrics

| WER | MER | WIL | CER | LEV-NORM | Completeness | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |
|-----|-----|-----|-----|----------|--------------|-------------------|---------------|--------------------|------------------|------------------|
| 0.1471 | 0.1460 | 0.2136 | 0.0927 | 0.0839 | 0.9554 | 0.0000 | 0.0000 | NA | 1.0000 | 1.0000 |


*Lower values indicate better OCR performance (except Completeness where 1.0 is ideal). Structural metrics range from 0.0–1.0 (higher is better).*

## 🔍 Top Word Mismatches

| Ground Truth | OCR Output | Count |
|---|---|---:|
| `●` | `-` | **8** |
| `a` | `<MISSING>` | **2** |
| `#` | `<MISSING>` | **1** |
| `What` | `<MISSING>` | **1** |
| `is` | `<MISSING>` | **1** |
| `LLM` | `<MISSING>` | **1** |
| `as` | `<MISSING>` | **1** |
| `judge?` | `<MISSING>` | **1** |
| `TL;DR:` | `<MISSING>` | **1** |
| `deﬁned` | `defined` | **1** |