# 📊 OCR Evaluation Report

**🤖 Model:** `Surya – OCR text extraction`  
**📄 Ground Truth:** `ตอบข้อสอบถ_161025_2_3Pages-pages-1.md (Ground Truth)`  
**📝 OCR Output:** `ตอบข้อสอบถ_161025_2_3Pages-pages-1.pdf (OCR Output)`  
**🕒 Timestamp:** 2026-01-14 10:31:57  

## 📊 Unified Metrics

| WER | MER | WIL | CER | LEV-DIST (chars) | Completeness | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |
|-----|-----|-----|-----|-------------------|--------------|-------------------|---------------|--------------------|------------------|------------------|
| 0.5294 | 0.5294 | 0.7104 | 0.0480 | 102 | 0.9656 | 0.0000 | 1.0000 | NA | 1.0000 | 0.5000 |


*Lower values indicate better OCR performance (except Completeness where 1.0 is ideal). Structural metrics range from 0.0–1.0 (higher is better).*

## 🔍 Top Word Mismatches

| Ground Truth | OCR Output | Count |
|---|---|---:|
| `**ข้อสอบถามที่` | `ข้อสอบถามที่` | **5** |
| `###` | `<MISSING>` | **2** |
| `๐๐๒๐/๑๙๒๘๐` | `๐๐๒๐/๑๙` | **1** |
| `**สำนักงานคณะกรรมการการเลือกตั้ง**` | `หนึ่ง` | **1** |
| `**ศูนย์ราชการเฉลิมพระเกียรติฯ**` | `สำนักงานคณะกรรมการการเลือกตั้ง` | **1** |
| `**ถนนแจ้งวัฒนะ` | `ศูนย์ราชการเฉลิมพระเกียรติฯ` | **1** |
| `เขตหลักสี่**` | `ถนนแจ้งวัฒนะ` | **1** |
| `**กรุงเทพฯ` | `เขตหลักสี่` | **1** |
| `๑๐๒๑๐**` | `กรุงเทพฯ` | **1** |
| `**19` | `๑๐๒๑๐` | **1** |