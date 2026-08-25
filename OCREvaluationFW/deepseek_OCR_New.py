"""
DeepSeek OCR module for extracting text from PDF files using DeepInfra API.
This module is designed to be a drop-in replacement for other OCR models
like Marker and Surya in the evaluation framework.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from openai import OpenAI, APIError, RateLimitError
except ImportError:
    raise ImportError(
        "OpenAI package is required for DeepSeek OCR. Install it with: pip install openai"
    )

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError(
        "PyMuPDF is required for PDF rendering. Install it with: pip install PyMuPDF"
    )

try:
    from PIL import Image
except ImportError:
    raise ImportError(
        "Pillow is required for image processing. Install it with: pip install Pillow"
    )


@dataclass
class PageResult:
    """Result for a single page extraction."""
    page_index: int
    text: str


def _image_to_base64_url(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URL."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    b64_string = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64_string}"


def _extract_text_from_image(
    client: OpenAI,
    image: Image.Image,
    model: str,
    max_tokens: int,
    temperature: float,
    prompt: str,
) -> str:
    """
    Extract text from a single image using DeepSeek OCR.
    """
    image_url = _image_to_base64_url(image)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
    except RateLimitError:
        print("Rate limit exceeded. Please wait and retry.")
        return "[Error: Rate limit exceeded]"
    except APIError as e:
        print(f"API error occurred: {e}")
        return f"[Error: API error occurred: {e}]"
    except Exception as e:
        print(f"Unexpected error: {e}")
        return f"[Error: Unexpected error: {e}]"


def ocr_pdf(
    pdf_bytes: bytes,
    settings,
    *,
    mode: str = "markdown",
    max_pages: Optional[int] = None,
    dpi: int = 300,
    custom_prompt: Optional[str] = None,
) -> Tuple[str, List[PageResult]]:
    """
    Extract text from PDF using DeepSeek OCR via DeepInfra.
    """
    # Set API token from environment if not in settings
    api_token = settings.api_token or os.environ.get("DEEPINFRA_TOKEN")
    if not api_token:
        raise ValueError("DeepInfra API token is not set. Provide it in config.toml or as DEEPINFRA_TOKEN environment variable.")

    client = OpenAI(
        api_key=api_token,
        base_url=settings.base_url,
    )

    if custom_prompt:
        prompt = custom_prompt
    elif mode == "markdown":
        prompt = (
            "Extract every single piece of text from the image into Markdown format. "
            "CRITICAL INSTRUCTIONS: "
            "1. START with the very top-most text (running headers, page numbers, large titles). Do NOT skip the header or title. "
            "2. Transcribe ALL headings, subheadings, and body text exactly as they appear. "
            "3. If there is a title at the top, it MUST be included as a # Heading 1. "
            "4. Do not summarize or omit anything. We need a full verbatim transcription. "
            "5. Preserve the structure (lists, tables, etc.) and reading order."
        )
    else:
        prompt = (
            "Perform a complete verbatim OCR of the image text. "
            "Start from the very top pixel and include headers, titles, and page numbers. "
            "End at the very bottom pixel including footers. "
            "Do not omit a single word."
        )

    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {str(e)}")

    total_pages = len(pdf_document)
    num_pages_to_process = min(max_pages, total_pages) if max_pages and max_pages > 0 else total_pages

    images = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    try:
        for page_num in range(num_pages_to_process):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
    except Exception as e:
        pdf_document.close()
        raise RuntimeError(f"Failed to convert PDF pages to images: {str(e)}")
    finally:
        pdf_document.close()

    per_page_results: List[PageResult] = []
    combined_parts: List[str] = []

    for page_idx, page_image in enumerate(images):
        page_num = page_idx + 1
        
        page_text = _extract_text_from_image(
            client=client,
            image=page_image,
            model=settings.model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            prompt=prompt,
        )

        if not page_text.strip():
            page_text = f"[No text extracted from page {page_num}]"

        if len(images) > 1:
            page_separator = f"\\n\\n---\\n**Page {page_num}**\\n---\\n\\n"
            combined_parts.append(page_separator + page_text.strip())
        else:
            combined_parts.append(page_text.strip())
            
        per_page_results.append(
            PageResult(page_index=page_idx, text=page_text.strip())
        )

    combined_text = "\\n\\n".join(combined_parts)
    
    return combined_text, per_page_results
