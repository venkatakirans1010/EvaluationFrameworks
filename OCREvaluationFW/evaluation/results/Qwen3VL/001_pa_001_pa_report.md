# 📊 OCR Evaluation Report

**🤖 Model:** `Qwen3 VL 8B – via OpenRouter`  
**📄 Ground Truth:** `001_page_1-1.md`  
**📝 OCR Output:** `001_page_1-1.pdf`  
**🕒 Timestamp:** 2026-01-22 16:10:34  

## 📊 Unified Metrics

| WER | MER | WIL | CER | LEV-DIST (chars) | Completeness | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |
|-----|-----|-----|-----|-------------------|--------------|-------------------|---------------|--------------------|------------------|------------------|
| 0.0993 | 0.0985 | 0.1814 | 0.0287 | 49 | 1.0000 | 1.0000 | 0.0000 | NA | 1.0000 | 1.0000 |


*Lower values indicate better OCR performance (except Completeness where 1.0 is ideal). Structural metrics range from 0.0–1.0 (higher is better).*

## 🔍 Top Word Mismatches

| Ground Truth | OCR Output | Count |
|---|---|---:|
| `●` | `-` | **8** |
| `deﬁned` | `defined` | **1** |
| `evaluate` | `*evaluate` | **1** |
| `quality.` | `quality*.` | **1** |
| `you’re` | `you're` | **1** |
| `workﬂows.` | `workflows.` | **1** |
| `deﬁne.` | `define.` | **1** |
| `Politeness:` | `**Politeness**:` | **1** |
| `Bias:` | `**Bias**:` | **1** |
| `Tone:` | `**Tone**:` | **1** |