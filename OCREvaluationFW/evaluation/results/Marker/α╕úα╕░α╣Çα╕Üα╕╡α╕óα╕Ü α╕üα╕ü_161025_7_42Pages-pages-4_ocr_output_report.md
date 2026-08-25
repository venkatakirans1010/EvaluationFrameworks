# 📊 OCR Evaluation Report

**🤖 Model:** `Marker – Convert to Markdown/HTML/JSON`  
**📄 Ground Truth:** `α╕úα╕░α╣Çα╕Üα╕╡α╕óα╕Ü α╕üα╕ü_161025_7_42Pages-pages-4.md (Ground Truth)`  
**📝 OCR Output:** `α╕úα╕░α╣Çα╕Üα╕╡α╕óα╕Ü α╕üα╕ü_161025_7_42Pages-pages-4.pdf (OCR Output)`  
**🕒 Timestamp:** 2026-01-27 10:10:14  

## 📊 Unified Metrics

| WER | MER | WIL | CER | LEV-DIST (chars) | Completeness | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |
|-----|-----|-----|-----|-------------------|--------------|-------------------|---------------|--------------------|------------------|------------------|
| 1.1791 | 0.8404 | 0.9643 | 0.3621 | 474 | 1.0000 | 0.0000 | 1.0000 | NA | 1.0000 | 1.0000 |


*Lower values indicate better OCR performance (except Completeness where 1.0 is ideal). Structural metrics range from 0.0–1.0 (higher is better).*

## 🔍 Top Word Mismatches

| Ground Truth | OCR Output | Count |
|---|---|---:|
| `ชื่อ...ชื่อสกุล...` | `ชื่อ.....` | **4** |
| `ชื่อ...ชื่อสกุล...อายุ` | `ชื่อสกุล.....` | **4** |
| `...` | `อายุ.....ปี` | **4** |
| `สัญชาติ...` | `ชื่อสกุล.....` | **3** |
| `จังหวัด...` | `จังหวัด.....` | **3** |
| `ประเทศ...` | `ประเทศ.....` | **3** |
| `อายุ...ปี` | `อายุ.....ปี` | **3** |
| `ปี` | `ชื่อ.....` | **3** |
| `พรรค...` | `พรรค.....` | **2** |
| `วันที่สมัคร...` | `วันที่สมัคร.....` | **2** |