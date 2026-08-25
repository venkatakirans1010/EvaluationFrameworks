from __future__ import annotations
import sys
import io

try:
    import fitz  # PyMuPDF
except Exception as e:
    print(f"PyMuPDF not available: {e}")
    sys.exit(1)

from config import get_openrouter_settings
from qwen3_vl_openrouter import ocr_pdf


def make_tiny_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4-ish
    text = (
        "Hello from the Qwen3 VL smoke test!\n"
        "This is a tiny PDF generated to verify end-to-end OCR via OpenRouter.\n"
        "Numbers: 12345, Symbols: # * - _ \n"
    )
    page.insert_text((72, 96), text, fontsize=14)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def main() -> int:
    try:
        settings = get_openrouter_settings()
    except Exception as e:
        print(f"Failed to load OpenRouter settings: {e}")
        return 2

    pdf_bytes = make_tiny_pdf_bytes()
    try:
        combined, pages = ocr_pdf(
            pdf_bytes=pdf_bytes,
            settings=settings,
            mode="plain",
            max_pages=1,
            dpi=144,
            custom_prompt=None,
        )
        preview = (combined or "").strip().replace("\r", "")
        print("=== Qwen3 VL Smoke Test Output (first 400 chars) ===")
        print(preview[:400])
        print("\n=== Pages ===", len(pages))
        return 0
    except Exception as e:
        print(f"Qwen smoke test failed: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
