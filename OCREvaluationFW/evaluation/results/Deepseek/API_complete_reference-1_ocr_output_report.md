# 📊 OCR Evaluation Report

**🤖 Model:** `Deepseek – OCR via DeepInfra`  
**📄 Ground Truth:** `API_complete_reference-1.md (Ground Truth)`  
**📝 OCR Output:** `API_complete_reference-1.pdf (OCR Output)`  
**🕒 Timestamp:** 2025-12-29 23:22:56  

## 📊 Unified Metrics

| WER | MER | WIL | CER | LEV-NORM | Completeness | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |
|-----|-----|-----|-----|----------|--------------|-------------------|---------------|--------------------|------------------|------------------|
| 0.1327 | 0.1307 | 0.2036 | 0.7684 | 0.0503 | 0.9837 | 0.0000 | 1.0000 | NA | 1.0000 | 1.0000 |


*Lower values indicate better OCR performance (except Completeness where 1.0 is ideal). Structural metrics range from 0.0–1.0 (higher is better).*

## 🔍 Top Word Mismatches

| Ground Truth | OCR Output | Count |
|---|---|---:|
| `•` | `-` | **10** |
| `Testing:**` | `Testing**:` | **2** |
| `Enhances` | `**Enhances` | **2** |
| `End` | `0` | **1** |
| `Ensures` | `**Ensures` | **1** |
| `Functionality` | `Functionality**` | **1** |
| `Requirements:**` | `Requirements**:` | **1** |
| `Early:**` | `Early**:` | **1** |
| `Improves` | `**Improves` | **1** |
| `Reliability` | `Reliability**` | **1** |