"""
Qwen3 VL 8B Instruct OCR module via OpenRouter.
Renders PDF pages to images and sends them to OpenRouter's
OpenAI-compatible chat completions API with vision inputs.

This module is fully self-contained to avoid interfering with existing models.
"""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import requests
from requests import RequestException

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError(
        "PyMuPDF is required for PDF rendering. Install it with: pip install PyMuPDF"
    )

try:
    from PIL import Image, ImageOps, ImageFilter
except ImportError:
    raise ImportError(
        "Pillow is required for image processing. Install it with: pip install Pillow"
    )

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except Exception:
        tomllib = None  # optional; we can rely on env vars


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_ID = "qwen/qwen3-vl-8b-instruct"  # Replace with the exact model id from OpenRouter


@dataclass
class PageResult:
    page_index: int
    text: str


@dataclass
class OpenRouterSettings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL_ID
    max_tokens: int = 4096
    temperature: float = 0.1
    site_url: Optional[str] = None
    site_title: Optional[str] = None


def _read_openrouter_settings_from_toml(path: str = "config.toml") -> Optional[OpenRouterSettings]:
    if tomllib is None:
        return None
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        section = data.get("openrouter")
        if not section:
            return None
        api_key = str(section.get("api_key", "")).strip()
        if not api_key:
            return None
        return OpenRouterSettings(
            api_key=api_key,
            base_url=str(section.get("base_url", DEFAULT_BASE_URL)),
            model=str(section.get("model", DEFAULT_MODEL_ID)),
            max_tokens=int(section.get("max_tokens", 4096)),
            temperature=float(section.get("temperature", 0.1)),
            site_url=(str(section.get("site_url")) if section.get("site_url") else None),
            site_title=(str(section.get("site_title")) if section.get("site_title") else None),
        )
    except Exception:
        return None


def get_openrouter_settings() -> OpenRouterSettings:
    """Resolve OpenRouter settings from env or config.toml.

    Priority:
    1. Env `OPENROUTER_API_KEY` (and optional `OPENROUTER_MODEL`, `OPENROUTER_BASE_URL`)
    2. `[openrouter]` section in config.toml
    3. Raise if no API key found
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    base_url = os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    site_url = os.environ.get("OPENROUTER_SITE_URL")
    site_title = os.environ.get("OPENROUTER_SITE_TITLE")

    if api_key:
        return OpenRouterSettings(
            api_key=api_key,
            base_url=base_url,
            model=model,
            site_url=site_url,
            site_title=site_title,
        )
    # Fallback to TOML
    toml_settings = _read_openrouter_settings_from_toml()
    if toml_settings:
        return toml_settings
    raise ValueError(
        "OpenRouter API key is not set. Provide it via environment variable OPENROUTER_API_KEY or in config.toml under [openrouter]."
    )


def _image_to_base64_url(image: Image.Image) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_string}"


def _enhance_image(image: Image.Image, enable: bool = False) -> Image.Image:
    """Optionally enhance the image for OCR.

    - Convert to RGB
    - Autocontrast
    - Mild sharpening
    """
    if not enable:
        return image
    img = image
    if img.mode != "RGB":
        img = img.convert("RGB")
    try:
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.SHARPEN)
    except Exception:
        # If anything goes wrong, just return the original to avoid breaking flow
        return image
    return img


def _tile_image(image: Image.Image, grid: Tuple[int, int] = (2, 2), overlap: float = 0.05) -> List[Image.Image]:
    """Split the image into a grid of tiles with slight overlap to preserve characters at boundaries.

    Returns tiles ordered top-to-bottom, left-to-right.
    """
    rows, cols = max(1, grid[0]), max(1, grid[1])
    w, h = image.size
    tiles: List[Image.Image] = []
    x_step = w / cols
    y_step = h / rows
    x_ov = int(x_step * overlap)
    y_ov = int(y_step * overlap)
    for r in range(rows):
        for c in range(cols):
            left = int(max(0, c * x_step - (x_ov if c > 0 else 0)))
            upper = int(max(0, r * y_step - (y_ov if r > 0 else 0)))
            right = int(min(w, (c + 1) * x_step + (x_ov if c < cols - 1 else 0)))
            lower = int(min(h, (r + 1) * y_step + (y_ov if r < rows - 1 else 0)))
            tiles.append(image.crop((left, upper, right, lower)))
    return tiles


def _extract_text_from_images(
    images: List[Image.Image],
    model: str,
    max_tokens: int,
    temperature: float,
    prompt: str,
    api_key: str,
    base_url: str,
    extra_headers: Optional[Dict[str, str]] = None,
    system_instructions: Optional[str] = None,
) -> str:
    contents: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images:
        image_url = _image_to_base64_url(img)
        contents.append({"type": "image_url", "image_url": {"url": image_url}})

    messages: List[Dict[str, Any]] = []
    if system_instructions:
        messages.append({"role": "system", "content": system_instructions})
    messages.append({"role": "user", "content": contents})
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")) or ""
    except RequestException as e:
        return f"[Error: HTTP request failed: {e}]"
    except Exception as e:
        return f"[Error: Unexpected error: {e}]"


def _extract_text_from_image_single(
    image: Image.Image,
    model: str,
    max_tokens: int,
    temperature: float,
    prompt: str,
    api_key: str,
    base_url: str,
    extra_headers: Optional[Dict[str, str]] = None,
    system_instructions: Optional[str] = None,
) -> str:
    image_url = _image_to_base64_url(image)
    messages: List[Dict[str, Any]] = []
    if system_instructions:
        messages.append({"role": "system", "content": system_instructions})
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    })

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")) or ""
    except RequestException as e:
        return f"[Error: HTTP request failed: {e}]"
    except Exception as e:
        return f"[Error: Unexpected error: {e}]"


def ocr_pdf(
    pdf_bytes: bytes,
    settings: OpenRouterSettings,
    *,
    mode: str = "markdown",
    max_pages: Optional[int] = None,
    dpi: int = 300,
    custom_prompt: Optional[str] = None,
    # New optional enhancements (safe defaults)
    language_hint: Optional[str] = None,
    layout: str = "auto",  # one of: auto, single, two_column
    enhance_image: bool = False,
    highres_tiling: bool = False,
    tile_grid: Tuple[int, int] = (2, 2),
    tile_overlap: float = 0.05,
    temperature_override: Optional[float] = None,
    multi_image_single_request: bool = True,
) -> Tuple[str, List[PageResult]]:
    """Extract text from a PDF using Qwen3 VL 8B via OpenRouter.

    Returns combined markdown/plain text and per-page results.
    """
    if not settings.api_key:
        raise ValueError("OpenRouter API key missing.")

    # Build optional OpenRouter headers
    extra_headers: Dict[str, str] = {}
    if settings.site_url:
        extra_headers["HTTP-Referer"] = settings.site_url
    if settings.site_title:
        extra_headers["X-Title"] = settings.site_title

    layout_clause = (
        "For two-column layouts, read the entire left column from top to bottom, then the right column from top to bottom."
        if layout == "two_column"
        else ("Assume a single reading order, top-to-bottom, left-to-right." if layout == "single" else "Follow the natural reading order; if two-column, read left column first then right.")
    )
    lang_clause = (
        f"Output MUST stay in the original language; do not translate or romanize. Language hint: {language_hint}."
        if language_hint
        else "Output MUST stay in the original language; do not translate or romanize."
    )

    guard_clause = (
        "Return only the transcription; do not add explanations or summaries. If any portion is unreadable, output [UNREADABLE] exactly. Do NOT translate."
    )
    if custom_prompt:
        prompt = custom_prompt
    elif mode == "markdown":
        prompt = (
            "Extract every piece of text as Markdown, preserving structure (headings, lists, tables). "
            "Start with top-most headers and titles, include page numbers if visible. "
            f"{layout_clause} "
            f"{lang_clause} "
            f"{guard_clause}"
        )
    else:
        prompt = (
            "Perform a complete verbatim OCR transcription of the image. "
            "Include headers, titles, page numbers, footers, and all body text. "
            f"{layout_clause} "
            f"{lang_clause} "
            f"{guard_clause}"
        )

    system_instructions = (
        "You are an OCR transcription assistant. Return only the transcription in the requested format. "
        "Do not add commentary. Do not translate. Preserve original punctuation, numerals, and diacritics."
    )

    # Render PDF pages
    try:
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}")

    total_pages = len(pdf_doc)
    num_pages = min(max_pages, total_pages) if max_pages and max_pages > 0 else total_pages

    images: List[Image.Image] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    try:
        for page_num in range(num_pages):
            page = pdf_doc[page_num]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = _enhance_image(img, enable=enhance_image)
            images.append(img)
    except Exception as e:
        pdf_doc.close()
        raise RuntimeError(f"Failed to convert PDF pages to images: {e}")
    finally:
        pdf_doc.close()

    per_page_results: List[PageResult] = []
    combined_parts: List[str] = []

    for idx, img in enumerate(images):
        page_num = idx + 1
        page_images: List[Image.Image] = [img]
        if highres_tiling:
            try:
                page_images = _tile_image(img, grid=tile_grid, overlap=tile_overlap)
            except Exception:
                page_images = [img]

        # Use either a single request with multiple images, or sequential one-image requests
        temp_value = settings.temperature if temperature_override is None else float(temperature_override)
        if multi_image_single_request:
            text = _extract_text_from_images(
                images=page_images,
                model=settings.model,
                max_tokens=settings.max_tokens,
                temperature=temp_value,
                prompt=prompt,
                api_key=settings.api_key,
                base_url=settings.base_url,
                extra_headers=extra_headers or None,
                system_instructions=system_instructions,
            )
        else:
            parts: List[str] = []
            for tile in page_images:
                t = _extract_text_from_image_single(
                    image=tile,
                    model=settings.model,
                    max_tokens=settings.max_tokens,
                    temperature=temp_value,
                    prompt=prompt,
                    api_key=settings.api_key,
                    base_url=settings.base_url,
                    extra_headers=extra_headers or None,
                    system_instructions=system_instructions,
                )
                parts.append(t.strip() or "")
            text = "\n".join(p for p in parts if p)
        if not text.strip():
            text = f"[No text extracted from page {page_num}]"
        if len(images) > 1:
            sep = f"\n\n---\n**Page {page_num}**\n---\n\n"
            combined_parts.append(sep + text.strip())
        else:
            combined_parts.append(text.strip())
        per_page_results.append(PageResult(page_index=idx, text=text.strip()))

    combined_text = "\n\n".join(combined_parts)
    return combined_text, per_page_results
