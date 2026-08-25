# 📊 OCR Evaluation Report

**🤖 Model:** `Qwen3 VL 8B – via OpenRouter`  
**📄 Ground Truth:** `class-10-sample-paper-2020-21-telugu-pages-2.md (Ground Truth)`  
**📝 OCR Output:** `class-10-sample-paper-2020-21-telugu-pages-2.pdf (OCR Output)`  
**🕒 Timestamp:** 2026-01-22 10:49:31  

## 📊 Unified Metrics

| WER | MER | WIL | CER | LEV-DIST (chars) | Completeness | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |
|-----|-----|-----|-----|-------------------|--------------|-------------------|---------------|--------------------|------------------|------------------|
| 1.0335 | 0.9720 | 0.9992 | 0.8762 | 2102 | 1.0000 | 1.0000 | 0.0000 | NA | 1.0000 | 1.0000 |


*Lower values indicate better OCR performance (except Completeness where 1.0 is ideal). Structural metrics range from 0.0–1.0 (higher is better).*

## 🔍 Top Word Mismatches

| Ground Truth | OCR Output | Count |
|---|---|---:|
| `[Image:` | `<MISSING>` | **1** |
| `From:` | `From` | **1** |
| `http://cbseportal.com/]` | `:` | **1** |
| `భారతీయ` | `http://cbseportal.com/` | **1** |
| `సంస్కృతి` | `ప్రాథమిక` | **1** |
| `అత్యంత` | `సంస్కృతి` | **1** |
| `ప్రాచీనమైనది.` | `మాత్రమే` | **1** |
| `మహారాజులు` | `ప్రాథమిక` | **1** |
| `రామాయణాది` | `సంస్కృతి` | **1** |
| `గ్రంథాలను` | `అభివృద్ధి` | **1** |