"""
DeepSeek OCR module for extracting text from PDF files using DeepInfra API.
This module integrates with the OCR evaluation framework, providing
functionality similar to Marker and Surya models.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from openai import OpenAI
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


def _crop_header_footer(
    image: Image.Image,
    header_ratio: float = 0.12,
    footer_ratio: float = 0.12,
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """
    Crop header and footer regions from the image.
    
    Args:
        image: PIL Image object
        header_ratio: Fraction of page height for header (e.g., 0.12 = 12%)
        footer_ratio: Fraction of page height for footer (e.g., 0.12 = 12%)
    
    Returns:
        Tuple of (header_crop, footer_crop), each may be None if ratio is 0
    """
    width, height = image.size
    header_crop = None
    footer_crop = None
    
    if header_ratio > 0:
        header_height = int(height * header_ratio)
        if header_height > 0:
            header_crop = image.crop((0, 0, width, header_height))
    
    if footer_ratio > 0:
        footer_height = int(height * footer_ratio)
        if footer_height > 0:
            footer_crop = image.crop((0, height - footer_height, width, height))
    
    return header_crop, footer_crop


def _image_to_base64_url(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URL."""
    # Ensure image is in RGB mode for best compatibility
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
    system_prompt: str,
) -> str:
    """
    Extract text from a single image using DeepSeek OCR via OpenAI-compatible API.
    
    Args:
        client: OpenAI client instance
        image: PIL Image object
        model: Model name (e.g., "deepseek-ai/DeepSeek-OCR")
        max_tokens: Maximum tokens for response
        temperature: Sampling temperature
        system_prompt: System prompt for extraction
    
    Returns:
        Extracted text
    """
    image_url = _image_to_base64_url(image)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    return response.choices[0].message.content or ""


def ocr_pdf(
    pdf_bytes: bytes,
    settings,
    *,
    mode: str = "markdown",
    max_pages: Optional[int] = None,
    batch_size: int = 1,
    dpi: int = 300,
    preserve_headers_footers: bool = True,
    custom_prompt: Optional[str] = None,
    header_ratio: float = 0.12,
    footer_ratio: float = 0.12,
) -> Tuple[str, List[PageResult]]:
    """
    Extract text from PDF using DeepSeek OCR via DeepInfra.
    
    Args:
        pdf_bytes: PDF file as bytes
        settings: DeepInfraSettings from config
        mode: Extraction mode ("markdown" or "plain")
        max_pages: Maximum pages to process (None = all)
        batch_size: Pages per API call (currently processes one at a time)
        dpi: DPI for PDF rendering
        preserve_headers_footers: Include headers/footers in output
        custom_prompt: Custom system prompt (overrides default)
        header_ratio: Height of header band as fraction of page height
        footer_ratio: Height of footer band as fraction of page height
    
    Returns:
        Tuple of (combined_text, per_page_results)
    """
    # Initialize OpenAI client with DeepInfra settings
    client = OpenAI(
        api_key=settings.api_token,
        base_url=settings.base_url,
    )
    
    # Prepare system prompt
    if custom_prompt:
        system_prompt = custom_prompt
    elif mode == "markdown":
        system_prompt = (
            "Extract ALL text from the image, preserving layout and structure in Markdown format. "
            "IMPORTANT: Include large titles and headings at the top of the page - do not skip them. "
            "Capture all headings (use # for main titles, ## for subtitles), lists, tables, page numbers, and formatting. "
            "Maintain the complete document hierarchy from top to bottom."
        )
    else:
        system_prompt = "Extract ALL text from the image as plain text, including titles and headings at the top. Preserve reading order from top to bottom."
    
    # Open PDF with PyMuPDF
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {str(e)}")
    
    # Determine page range
    total_pages = len(pdf_document)
    if max_pages and max_pages > 0:
        num_pages = min(max_pages, total_pages)
    else:
        num_pages = total_pages
    
    # Convert pages to images using PyMuPDF
    images = []
    zoom = dpi / 72.0  # Convert DPI to zoom factor (72 is default DPI)
    matrix = fitz.Matrix(zoom, zoom)
    
    try:
        for page_num in range(num_pages):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=matrix, alpha=False)  # No alpha channel
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            # Ensure RGB mode
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
    except Exception as e:
        pdf_document.close()
        raise RuntimeError(f"Failed to convert PDF pages to images: {str(e)}")
    finally:
        pdf_document.close()
    
    # Process each page
    per_page_results: List[PageResult] = []
    combined_parts: List[str] = []
    
    for page_idx, page_image in enumerate(images):
        page_num = page_idx + 1
        
        # Extract full page content
        try:
            page_text = _extract_text_from_image(
                client=client,
                image=page_image,
                model=settings.model,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                system_prompt=system_prompt,
            )
            
            if not page_text.strip():
                page_text = f"[No text extracted from page {page_num}]"
            
        except Exception as e:
            print(f"Warning: Failed to extract content for page {page_num}: {e}")
            page_text = f"[Error extracting page {page_num}: {str(e)}]"
        
        # Add page separator for multi-page documents
        if len(images) > 1:
            page_separator = f"\n\n---\n**Page {page_num}**\n---\n\n"
            combined_parts.append(page_separator + page_text.strip())
        else:
            combined_parts.append(page_text.strip())
        
        # Store per-page result
        per_page_results.append(
            PageResult(page_index=page_idx, text=page_text.strip())
        )
    
    # Combine all pages
    combined_text = "\n\n".join(combined_parts)
    
    return combined_text, per_page_results
