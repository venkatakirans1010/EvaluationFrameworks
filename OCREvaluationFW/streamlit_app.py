from __future__ import annotations

import base64
import time
from typing import Any, Dict, Iterable, List, Optional
import io
from datetime import datetime, timezone
import difflib
import json
import os
import socket
import shutil

import requests
from requests import exceptions as requests_exceptions
import streamlit as st

from config import get_settings, get_deepinfra_settings, get_openrouter_settings
from deepseek_OCR_New import ocr_pdf as deepseek_ocr_pdf
from qwen3_vl_openrouter import ocr_pdf as qwen_ocr_pdf

# Enhanced sidebar styling for professional appearance
st.markdown(
    """
    <style>
    /* Sidebar container */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: none;
        box-shadow: 4px 0 12px rgba(0, 0, 0, 0.15);
        /* Fit the width to the longest option text */
        width: fit-content !important;
        min-width: 260px !important; /* sensible minimum for short labels */
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 24px;
        padding-bottom: 24px;
        padding-right: 16px; /* avoid clipping when width shrinks */
    }
    
    /* Ensure all text inside the navigation looks white by default */
    [data-testid="stSidebar"] * {
        color: #ffffffcc; /* slightly translucent for non-primary text */
    }
    
    /* Sidebar title */
    [data-testid="stSidebar"] h1 {
        color: #ffffff !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        margin-bottom: 8px !important;
        padding-bottom: 16px !important;
        border-bottom: 2px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Sidebar navigation label ("Navigation") */
    [data-testid="stSidebar"] label[data-baseweb="radio"] {
        color: #e5e7eb !important;
        font-weight: 500 !important;
        font-size: 0.80rem !important;
        text-transform: none !important;
        letter-spacing: 0.02em !important;
    }
    
    /* Radio button container */
    [data-testid="stSidebar"] [data-baseweb="radio"] {
        gap: 16px; /* increased spacing between options */
    }
    
    /* Individual radio option */
    [data-testid="stSidebar"] [data-baseweb="radio"] [role="radio"] {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        margin-bottom: 8px !important; /* revert: spacing comes from container gap */
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        color: #ffffff !important;
        backdrop-filter: blur(2px);
        -webkit-backdrop-filter: blur(2px);
        white-space: nowrap; /* keep labels on one line */
        display: flex; /* ensure icon and text are linked */
        align-items: center;
        gap: 6px; /* reduce space between radio icon and text */
    }
    
    /* Radio option text – force pure white */
    [data-testid="stSidebar"] [data-baseweb="radio"] [role="radio"] * {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
    }
    
    /* Hover state */
    [data-testid="stSidebar"] [data-baseweb="radio"] [role="radio"]:hover {
        background: rgba(59, 130, 246, 0.18) !important;
        border-color: rgba(59, 130, 246, 0.35) !important;
        transform: translateX(4px) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.18) !important;
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] [role="radio"]:hover * {
        color: #ffffff !important;
    }
    
    /* Selected state - enhanced highlighting */
    [data-testid="stSidebar"] [data-baseweb="radio"] [role="radio"][aria-checked="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        border: 2px solid #60a5fa !important;
        color: #ffffff !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.45), 
                    0 0 0 3px rgba(59, 130, 246, 0.18),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        transform: translateX(8px) !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] [role="radio"][aria-checked="true"] * {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Divider after sidebar navigation */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.12);
        margin: 24px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar header
st.sidebar.title("OCR Evaluation Framework")

# Left-side navigation menu (default to first option)
selected_menu = st.sidebar.radio(
    "Navigation",
    [
        "🚀 Auto Extract and Evaluate",
        "📦 Batch: Auto‑Map Ground Truth",
        "📋 Manual File Comparison",
    ],
    index=0
)
from multi_run_evaluation import MultiRunRunner, create_extraction_wrapper, get_pdf_basename
from multi_run_reporter import MultiRunReporter

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def _strip_image_descriptions(text: str) -> str:
    """Remove image tags and common caption/figure lines from text.
    - Removes Markdown images: ![alt](url)
    - Removes HTML <img> tags and <figcaption> blocks
    - If image tokens are present, removes lines like "Figure 1:", "Fig. 2:", "Caption:" etc.
    """
    import re as _re
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    # Remove explicit image markups
    text = _re.sub(r'!\[[^\]]*\]\([^\)]+\)', ' ', text)  # Markdown image
    text = _re.sub(r'<img\b[^>]*>', ' ', text, flags=_re.I)  # HTML <img>
    text = _re.sub(r'<figcaption\b[^>]*>.*?</figcaption>', ' ', text, flags=_re.I | _re.S)  # HTML figcaption
    # Remove caption-like lines only if image tokens exist
    has_image_tokens = ('![' in text) or ('<img' in text.lower()) or ('<figure' in text.lower())
    if has_image_tokens:
        caption_line_re = _re.compile(r'^\s*(?:Figure|Fig\.|Image|Illustration)\s*\d*\s*[:.\-]\s*.*$', _re.I)
        labelled_line_re = _re.compile(r'^\s*(?:Caption|Image\s*Description|Alt\s*Text|Photo\s*Caption)\s*[:\-]\s*.*$', _re.I)
        filtered_lines = []
        for line in text.splitlines():
            if caption_line_re.match(line) or labelled_line_re.match(line):
                continue
            filtered_lines.append(line)
        text = "\n".join(filtered_lines)
    return text


def _render_pdf(bytes_data: bytes, height: int = 800) -> None:
    """
    Render a PDF preview by converting pages to images.
    Falls back to basic info if conversion libraries are not available.
    """
    st.write("**PDF Preview:**")
    
    if PYMUPDF_AVAILABLE and PIL_AVAILABLE:
        try:
            # Convert PDF to images using PyMuPDF
            pdf_document = fitz.open(stream=bytes_data, filetype="pdf")
            
            # Display basic PDF info
            st.info(f"PDF: {pdf_document.page_count} pages, {len(bytes_data):,} bytes")
            
            # Convert and display first few pages as images
            max_pages_to_show = min(3, pdf_document.page_count)
            
            for page_num in range(max_pages_to_show):
                page = pdf_document[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(1.5, 1.5)  # Zoom factor for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Display the image
                st.image(
                    img_data,
                    caption=f"Page {page_num + 1}",
                    use_container_width=True
                )
            
            if pdf_document.page_count > max_pages_to_show:
                st.info(f"Showing first {max_pages_to_show} pages of {pdf_document.page_count} total pages.")
            
            pdf_document.close()
            return
            
        except Exception as e:
            st.warning(f"Could not convert PDF to images: {str(e)}")
    
    # Fallback: Try iframe approach
    try:
        encoded = base64.b64encode(bytes_data).decode("utf-8")
        pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{encoded}"
            width="100%"
            height="{height}"
            type="application/pdf"
            style="border: 1px solid #ccc; border-radius: 4px;">
            <p>Your browser does not support PDF preview.</p>
        </iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)
        st.info(f"PDF uploaded successfully ({len(bytes_data):,} bytes)")
        
    except Exception as e:
        # Final fallback: Just show file info
        st.info(f"PDF uploaded successfully ({len(bytes_data):,} bytes). Preview not available in this environment.")
        st.write("The PDF will be processed when you click 'Run with Datalab'.")


def _request_with_failover(
    primary: str,
    fallbacks: Iterable[str],
    files: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
) -> Dict[str, Any]:
    """
    Try the primary endpoint followed by any fallbacks. Use a short connect timeout
    so unreachable hosts fail fast. Accumulate errors and raise a clear RuntimeError
    if all endpoints fail.
    """
    tried_endpoints: List[str] = []
    last_error: Exception | None = None

    # Build a de-duplicated ordered list of endpoints, prioritizing api.datalab.to
    endpoints = list(dict.fromkeys([primary, *fallbacks]).keys())
    
    # Reorder to prioritize api.datalab.to if it exists
    api_endpoint = "https://api.datalab.to/api/v1/marker"
    ocr_api_endpoint = "https://api.datalab.to/api/v1/ocr"
    
    if api_endpoint in endpoints or ocr_api_endpoint in endpoints:
        # Move api.datalab.to to front if it exists
        reordered = []
        for ep in endpoints:
            if "api.datalab.to" in ep:
                reordered.insert(0, ep)
            else:
                reordered.append(ep)
        endpoints = reordered

    for endpoint in endpoints:
        if not endpoint:
            continue
        tried_endpoints.append(endpoint)

        # Use longer connect timeout for better reliability
        connect_timeout = 15.0
        req_timeout = (connect_timeout, timeout)

        try:
            print(f"Attempting Datalab request to {endpoint} (connect timeout={connect_timeout}s, read timeout={timeout}s)")
            response = requests.post(
                endpoint,
                files=files,
                headers=headers,
                timeout=req_timeout,
            )
            try:
                response.raise_for_status()
            except requests_exceptions.HTTPError as http_err:
                status_code = getattr(response, "status_code", None)
                # If it's an authentication/authorization issue, fail fast with a clear message
                if status_code in (401, 403):
                    detail = (
                        f"HTTP {status_code} from {endpoint}. Your API key may be missing, invalid, or not authorized for this endpoint. "
                        "Open config.toml and verify [datalab].api_key, and ensure the key has access to the selected service (Marker/OCR)."
                    )
                    # Raise a RuntimeError so callers can present a user-friendly message
                    raise RuntimeError(detail) from http_err
                # For other HTTP errors, keep trying fallbacks
                raise
            return response.json()
        except requests_exceptions.ConnectTimeout as exc:
            last_error = exc
            print(f"ConnectTimeout when contacting {endpoint}: {exc}")
            # Try next endpoint immediately
            continue
        except requests_exceptions.ReadTimeout as exc:
            last_error = exc
            print(f"ReadTimeout when contacting {endpoint}: {exc}")
            # Try next endpoint
            continue
        except requests_exceptions.ConnectionError as exc:
            last_error = exc
            print(f"ConnectionError when contacting {endpoint}: {exc}")
            continue
        except requests_exceptions.RequestException as exc:
            last_error = exc
            print(f"RequestException when contacting {endpoint}: {exc}")
            continue

    # If we reach here, all endpoints failed
    raise RuntimeError(
        "All configured Datalab endpoints failed. Tried: "
        + ", ".join(tried_endpoints)
        + (f". Last error: {last_error}" if last_error else "")
        + ". Try using the endpoint diagnostics to check connectivity."
    )


def _submit_marker_job(
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    settings,
    *,
    output_format: str,
    use_llm: bool,
    force_ocr: bool,
    max_pages: Optional[int],
    language: Optional[str] = None,
) -> Dict[str, Any]:
    files: Dict[str, Any] = {
        "file": (filename, file_bytes, mime_type),
        "output_format": (None, output_format),
        "paginate": (None, "true"),
        "use_llm": (None, str(use_llm).lower()),
        "force_ocr": (None, str(force_ocr).lower()),
    }
    if max_pages:
        files["max_pages"] = (None, str(max_pages))
    if language:
        files["language"] = (None, language)
    headers = {"X-API-Key": settings.api_key}
    response_payload = _request_with_failover(
        settings.endpoint,
        settings.fallback_endpoints,
        files,
        headers,
        settings.request_timeout_seconds,
    )
    if response_payload.get("request_check_url"):
        return _poll_until_complete(
            response_payload["request_check_url"], settings
        )
    return response_payload


def _submit_ocr_job(
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    settings,
    *,
    max_pages: Optional[int],
    language: Optional[str] = None,
) -> Dict[str, Any]:
    files: Dict[str, Any] = {
        "file": (filename, file_bytes, mime_type),
    }
    if max_pages:
        files["max_pages"] = (None, str(max_pages))
    if language:
        files["language"] = (None, language)
    headers = {"X-API-Key": settings.api_key}
    response_payload = _request_with_failover(
        settings.ocr_endpoint,
        settings.ocr_fallback_endpoints,
        files,
        headers,
        settings.request_timeout_seconds,
    )
    if response_payload.get("request_check_url"):
        return _poll_until_complete(
            response_payload["request_check_url"], settings
        )
    return response_payload


def _poll_until_complete(
    check_url: str, settings
) -> Dict[str, Any]:
    """
    Poll the provided check_url until the job status is 'complete' or we hit the deadline.
    This routine handles transient network errors by retrying until the poll timeout elapses.
    """
    headers = {"X-API-Key": settings.api_key}
    deadline = time.time() + settings.poll_timeout_seconds

    while time.time() < deadline:
        try:
            # Use a modest connect timeout so the poll loop does not hang on network issues
            connect_timeout = 10.0
            resp = requests.get(check_url, headers=headers, timeout=(connect_timeout, settings.request_timeout_seconds))
            resp.raise_for_status()
        except requests_exceptions.ConnectTimeout as exc:
            print(f"ConnectTimeout while polling {check_url}: {exc}. Retrying...")
            time.sleep(settings.poll_interval_seconds)
            continue
        except requests_exceptions.ReadTimeout as exc:
            print(f"ReadTimeout while polling {check_url}: {exc}. Retrying...")
            time.sleep(settings.poll_interval_seconds)
            continue
        except requests_exceptions.RequestException as exc:
            # Log and retry until the overall deadline
            print(f"Network error while polling {check_url}: {exc}. Retrying...")
            time.sleep(settings.poll_interval_seconds)
            continue

        try:
            payload = resp.json()
        except Exception as exc:
            print(f"Invalid JSON from {check_url}: {exc}. Retrying...")
            time.sleep(settings.poll_interval_seconds)
            continue

        status = payload.get("status")
        if status == "complete":
            if not payload.get("success", True):
                raise RuntimeError(payload.get("error", "Job failed."))
            return payload

        # Not complete yet — wait and poll again
        time.sleep(settings.poll_interval_seconds)

    raise TimeoutError(
        "Timed out waiting for Datalab job to finish. "
        "Try again with a smaller document or increase poll_timeout_seconds."
    )


def _submit_deepseek_job(
    *,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    settings,
    mode: str = "markdown",
    max_pages: Optional[int] = None,
    dpi: int = 300,
    custom_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Run DeepSeek OCR via DeepInfra and return a payload compatible with _extract_markdown().

    This function renders PDF pages locally and sends them as base64 data URLs
    to the DeepInfra OpenAI-compatible endpoint using the configuration in config.toml.
    """
    try:
        combined_text, per_page = deepseek_ocr_pdf(
            pdf_bytes=file_bytes,
            settings=settings,
            max_pages=max_pages,
            mode=mode,
            dpi=dpi,
            custom_prompt=custom_prompt,
        )
        # Build a payload structure that _extract_markdown can consume
        payload: Dict[str, Any] = {
            "markdown": combined_text,
            "result": {
                "model": settings.model,
                "outputs": [combined_text],
            },
            "pages": [
                {
                    "index": p.page_index,
                    "text_lines": [{"text": line} for line in (p.text or "").splitlines()],
                }
                for p in per_page
            ],
        }
        return payload
    except Exception as e:
        raise RuntimeError(f"DeepSeek OCR failed: {type(e).__name__}: {str(e)}")


def _submit_qwen_job(
    *,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    settings,
    mode: str = "markdown",
    max_pages: Optional[int] = None,
    dpi: int = 300,
    custom_prompt: Optional[str] = None,
    language_hint: Optional[str] = None,
    layout: str = "auto",
    enhance_image: bool = False,
    highres_tiling: bool = False,
    tile_rows: int = 2,
    tile_cols: int = 2,
    tile_overlap: float = 0.05,
    temperature_override: Optional[float] = None,
    multi_image_single_request: bool = True,
) -> Dict[str, Any]:
    """Run Qwen3 VL OCR via OpenRouter and return a payload compatible with _extract_markdown().

    Renders PDF pages locally and sends them as base64 data URLs to OpenRouter.
    """
    try:
        combined_text, per_page = qwen_ocr_pdf(
            pdf_bytes=file_bytes,
            settings=settings,
            max_pages=max_pages,
            mode=mode,
            dpi=dpi,
            custom_prompt=custom_prompt,
            language_hint=language_hint or None,
            layout=layout,
            enhance_image=enhance_image,
            highres_tiling=highres_tiling,
            tile_grid=(max(1, int(tile_rows)), max(1, int(tile_cols))),
            tile_overlap=tile_overlap,
            temperature_override=temperature_override,
            multi_image_single_request=multi_image_single_request,
        )
        payload: Dict[str, Any] = {
            "markdown": combined_text,
            "result": {
                "model": settings.model,
                "outputs": [combined_text],
            },
            "pages": [
                {
                    "index": p.page_index,
                    "text_lines": [{"text": line} for line in (p.text or "").splitlines()],
                }
                for p in per_page
            ],
        }
        return payload
    except Exception as e:
        raise RuntimeError(f"Qwen3 VL OCR failed: {type(e).__name__}: {str(e)}")


def _test_network_connectivity() -> Dict[str, Any]:
    """
    Comprehensive network connectivity test for Datalab endpoints.
    Returns detailed diagnostic information.
    """
    results = {
        "dns_resolution": {},
        "tcp_connectivity": {},
        "http_connectivity": {},
        "summary": {"total_tests": 0, "passed": 0, "failed": 0}
    }
    
    # Test endpoints
    test_hosts = [
        ("api.datalab.to", 443),
        ("marker.datalab.to", 443),
        ("www.datalab.to", 443)
    ]
    
    for hostname, port in test_hosts:
        results["summary"]["total_tests"] += 3  # DNS + TCP + HTTP
        
        # 1. DNS Resolution Test
        try:
            addr_info = socket.getaddrinfo(hostname, port)
            ip_addresses = [info[4][0] for info in addr_info]
            results["dns_resolution"][hostname] = {
                "status": "success",
                "ip_addresses": ip_addresses,
                "details": f"Resolved to {len(ip_addresses)} IP(s)"
            }
            results["summary"]["passed"] += 1
        except socket.gaierror as e:
            results["dns_resolution"][hostname] = {
                "status": "failed",
                "error": str(e),
                "details": "DNS resolution failed"
            }
            results["summary"]["failed"] += 1
            continue  # Skip TCP/HTTP tests if DNS fails
        except Exception as e:
            results["dns_resolution"][hostname] = {
                "status": "error",
                "error": str(e),
                "details": "Unexpected DNS error"
            }
            results["summary"]["failed"] += 1
            continue
        
        # 2. TCP Connectivity Test
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            result = sock.connect_ex((hostname, port))
            sock.close()
            
            if result == 0:
                results["tcp_connectivity"][hostname] = {
                    "status": "success",
                    "details": f"TCP connection to {hostname}:{port} successful"
                }
                results["summary"]["passed"] += 1
            else:
                results["tcp_connectivity"][hostname] = {
                    "status": "failed",
                    "error_code": result,
                    "details": f"TCP connection failed with code {result}"
                }
                results["summary"]["failed"] += 1
        except Exception as e:
            results["tcp_connectivity"][hostname] = {
                "status": "error",
                "error": str(e),
                "details": "TCP connection test failed"
            }
            results["summary"]["failed"] += 1
        
        # 3. HTTP Connectivity Test
        try:
            url = f"https://{hostname}/api/v1/marker"
            resp = requests.get(url, timeout=(5.0, 10.0))
            status_code = int(resp.status_code)
            
            if 200 <= status_code < 500:  # Any response is good (even 404)
                results["http_connectivity"][hostname] = {
                    "status": "success",
                    "status_code": status_code,
                    "details": f"HTTP response received (status {status_code})"
                }
                results["summary"]["passed"] += 1
            else:
                results["http_connectivity"][hostname] = {
                    "status": "failed",
                    "status_code": status_code,
                    "details": f"HTTP request failed with status {status_code}"
                }
                results["summary"]["failed"] += 1
        except requests_exceptions.ConnectTimeout as e:
            results["http_connectivity"][hostname] = {
                "status": "connect_timeout",
                "error": str(e),
                "details": "HTTP connection timed out"
            }
            results["summary"]["failed"] += 1
        except requests_exceptions.ConnectionError as e:
            results["http_connectivity"][hostname] = {
                "status": "connection_error",
                "error": str(e),
                "details": "HTTP connection error"
            }
            results["summary"]["failed"] += 1
        except Exception as e:
            results["http_connectivity"][hostname] = {
                "status": "error",
                "error": str(e),
                "details": "HTTP test failed"
            }
            results["summary"]["failed"] += 1
    
    return results


def _test_endpoints(settings) -> List[tuple]:
    """
    Test the primary endpoint and fallbacks for basic reachability.
    Returns a list of tuples: (endpoint, status, info)
    status: 'reachable' (HTTP code), 'status' (non-2xx HTTP), or error label ('connect_timeout', etc.)
    """
    endpoints = [settings.endpoint, *settings.fallback_endpoints]
    results: List[tuple] = []

    for ep in endpoints:
        if not ep:
            continue
        try:
            # Use a short connect timeout so unreachable hosts fail fast
            resp = requests.get(ep, timeout=(5.0, 10.0))
            status_code = getattr(resp, "status_code", None)
            if status_code is not None:
                status_code = int(status_code)
            if status_code and 200 <= status_code < 400:
                results.append((ep, "reachable", status_code))
            else:
                # Non-2xx/3xx responses are still useful to know (e.g., 404 means host reachable)
                results.append((ep, "status", status_code))
        except requests_exceptions.ConnectTimeout as exc:
            results.append((ep, "connect_timeout", str(exc)))
        except requests_exceptions.ReadTimeout as exc:
            results.append((ep, "read_timeout", str(exc)))
        except requests_exceptions.ConnectionError as exc:
            results.append((ep, "connection_error", str(exc)))
        except Exception as exc:
            results.append((ep, "error", str(exc)))

    return results


def _collect_text_blobs(node: Any) -> List[str]:
    """Recursively pull out textual content from varied API payloads."""
    if isinstance(node, str):
        return [node]

    if isinstance(node, dict):
        candidates: List[str] = []
        for key in (
            "markdown",
            "text",
            "content",
            "value",
            "raw",
            "data",
            "output",
        ):
            if key in node:
                candidates.extend(_collect_text_blobs(node[key]))
        return candidates

    if isinstance(node, Iterable):
        collected: List[str] = []
        for item in node:
            collected.extend(_collect_text_blobs(item))
        return collected

    return []


def _extract_markdown(payload: Dict[str, Any]) -> Optional[str]:
    """Normalize possible Marker / Chandra response structures."""
    top_level = payload.get("markdown")
    if isinstance(top_level, str) and top_level.strip():
        return top_level

    result = payload.get("result")
    if isinstance(result, dict):
        blobs = _collect_text_blobs(result.get("outputs"))
        if blobs:
            first = next((blob for blob in blobs if blob and blob.strip()), None)
            if first:
                return first

        for key in ("markdown", "html", "text"):
            if key in result and isinstance(result[key], str) and result[key].strip():
                return result[key]

    # OCR endpoint responses contain page-level text_lines.
    pages = payload.get("pages")
    if isinstance(pages, Iterable):
        lines: Iterable[str] = (
            (line.get("text", "") if isinstance(line, dict) else "")
            for page in pages
            if isinstance(page, dict)
            for line in page.get("text_lines", [])
        )
        joined = "\n".join(filter(None, lines))
        if joined.strip():
            return joined

    # Sometimes result["raw"] includes markdown blobs.
    raw_result = payload.get("raw")
    if isinstance(raw_result, str) and raw_result.strip():
        return raw_result

    return None


def render_side_by_side_diff(gt_text: str, ocr_text: str, gt_name: str = "Ground Truth", ocr_name: str = "OCR Output") -> str:
    """
    Render a compact side-by-side line-wise diff as HTML.
    Returns an HTML string that can be rendered with st.markdown(..., unsafe_allow_html=True).
    """
    import difflib
    import html

    gt_lines = gt_text.splitlines()
    ocr_lines = ocr_text.splitlines()
    matcher = difflib.SequenceMatcher(None, gt_lines, ocr_lines)

    left_html = []
    right_html = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                left_html.append(f'<div style="padding:6px 8px; line-height:1.4; border-left:3px solid #28a745;">{html.escape(gt_lines[i])}</div>')
            for j in range(j1, j2):
                right_html.append(f'<div style="padding:6px 8px; line-height:1.4; border-left:3px solid #28a745;">{html.escape(ocr_lines[j])}</div>')
        elif tag == "delete":
            for i in range(i1, i2):
                left_html.append(f'<div style="background:#ffeef0; padding:6px 8px; line-height:1.4; border-left:3px solid #dc3545; text-decoration:line-through;">{html.escape(gt_lines[i])}</div>')
        elif tag == "insert":
            for j in range(j1, j2):
                right_html.append(f'<div style="background:#d1f4d0; padding:6px 8px; line-height:1.4; border-left:3px solid #198754;">{html.escape(ocr_lines[j])}</div>')
        elif tag == "replace":
            for i in range(i1, i2):
                left_html.append(f'<div style="background:#fff3cd; padding:6px 8px; line-height:1.4; border-left:3px solid #ffc107;">{html.escape(gt_lines[i])}</div>')
            for j in range(j1, j2):
                right_html.append(f'<div style="background:#fff3cd; padding:6px 8px; line-height:1.4; border-left:3px solid #ffc107;">{html.escape(ocr_lines[j])}</div>')

    # Build two-column layout
    html_out = (
        '<div style="display:flex; gap:12px; align-items:flex-start; font-family: monospace;">'
        f'<div style="flex:1; min-width:40%; max-width:50%;">'
        f'<h4 style="margin:6px 0 8px 0;">{html.escape(gt_name)}</h4>'
        f'{"".join(left_html) if left_html else "<div style=\'padding:6px 8px;color:#6b7280;\'>No lines</div>"}'
        '</div>'
        f'<div style="flex:1; min-width:40%; max-width:50%;">'
        f'<h4 style="margin:6px 0 8px 0;">{html.escape(ocr_name)}</h4>'
        f'{"".join(right_html) if right_html else "<div style=\'padding:6px 8px;color:#6b7280;\'>No lines</div>"}'
        '</div>'
        '</div>'
    )
    return html_out


def main() -> None:
    settings = get_settings()
    st.set_page_config(
        page_title="OCR Evaluation Framework",
        layout="wide",
    )
    # Center-aligned title
    st.markdown(
        """
        <h1 style="text-align: center; margin-top: 0; margin-bottom: 0.5rem;">
            OCR Evaluation Framework
        </h1>
        <p style="text-align: center; color: #6b7280; font-size: 0.95rem; margin-bottom: 2rem;">
            Extract text from PDFs using Datalab API and evaluate against ground truth files
        </p>
        """,
        unsafe_allow_html=True
    )
    
    # Apply streamlined CSS for cleaner UI
    st.markdown(
        """
        <style>
        /* Global improvements */
        .stApp {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        /* Compact headers and spacing */
        .stMarkdown h1 {
            margin-bottom: 0.5rem !important;
        }
        
        .stMarkdown h2 {
            margin-top: 1.5rem !important;
            margin-bottom: 0.75rem !important;
            font-size: 1.5rem !important;
        }
        
        .stMarkdown h3 {
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
            font-size: 1.25rem !important;
        }
        
        /* Compact file uploaders */
        .stFileUploader > div {
            padding: 0.75rem !important;
        }
        
        /* Streamlined buttons */
        .stButton > button {
            border-radius: 6px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 500 !important;
        }
        
        .stDownloadButton > button {
            border-radius: 6px !important;
            padding: 0.4rem 0.8rem !important;
            font-size: 0.875rem !important;
        }
        
        /* Compact expanders */
        .streamlit-expanderHeader {
            font-size: 1rem !important;
            font-weight: 600 !important;
            padding: 0.5rem 0.75rem !important;
        }
        
        /* Cleaner metrics */
        [data-testid="metric-container"] {
            background-color: #f8fafc !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 6px !important;
            padding: 0.75rem !important;
        }
        
        /* Reduce column padding */
        .stColumn > div {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        
        /* Compact info boxes */
        .stAlert {
            padding: 0.75rem !important;
            margin: 0.5rem 0 !important;
        }
        
        /* Cleaner radio buttons */
        .stRadio > div {
            gap: 0.5rem !important;
        }
        
        /* Compact text inputs */
        .stTextInput > div > div > input {
            padding: 0.5rem 0.75rem !important;
        }
        
        /* Streamlined checkboxes */
        .stCheckbox {
            margin-bottom: 0.5rem !important;
        }
        
        /* Better spacing for sections */
        .main > div {
            padding-top: 1rem !important;
        }
        
        /* Compact captions */
        .stCaption {
            margin-bottom: 1rem !important;
        }
        
        /* Streamlined success/info messages */
        .stSuccess, .stInfo, .stWarning, .stError {
            padding: 0.5rem 0.75rem !important;
            margin: 0.5rem 0 !important;
        }
        
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ========================================
    # CONDITIONAL SECTION RENDERING
    # ========================================
    
    if selected_menu == "🚀 Auto Extract and Evaluate":
        render_auto_section(settings)
    elif selected_menu == "📦 Batch: Auto‑Map Ground Truth":
        render_batch_section(settings)
    elif selected_menu == "📋 Manual File Comparison":
        render_manual_section()


def render_auto_section(settings):
    """Render the Auto Extract and Evaluate section"""
    # ========================================
    # AUTO EXTRACT AND EVALUATE SECTION
    # ========================================
    st.header("🤖 Auto Extract and Evaluate")
    st.caption("Upload a PDF and optional ground truth file for automatic OCR extraction and evaluation")

    # Datalab endpoint diagnostics (moved inside Auto section)
    with st.expander("🔧 Datalab Endpoint Diagnostics", expanded=False):
        st.write(
            "Comprehensive network diagnostics for Datalab API connectivity issues. "
            "This tests DNS resolution, TCP connectivity, and HTTP responses."
        )
        
        col_diag1, col_diag2 = st.columns(2)
        
        with col_diag1:
            if st.button("🔍 Run Full Network Diagnostics", type="primary"):
                with st.spinner("Running comprehensive network tests..."):
                    try:
                        network_results = _test_network_connectivity()
                        
                        # Display summary
                        total = network_results["summary"]["total_tests"]
                        passed = network_results["summary"]["passed"]
                        failed = network_results["summary"]["failed"]
                        
                        if passed == total:
                            st.success(f"✅ All tests passed ({passed}/{total})")
                        elif passed > 0:
                            st.warning(f"⚠️ Partial connectivity ({passed}/{total} tests passed)")
                        else:
                            st.error(f"❌ All tests failed ({failed}/{total})")
                        
                        # Display detailed results
                        st.subheader("📊 Detailed Test Results")
                        
                        # DNS Results
                        st.markdown("**🌐 DNS Resolution:**")
                        for hostname, result in network_results["dns_resolution"].items():
                            if result["status"] == "success":
                                st.success(f"✅ {hostname}: {result['details']}")
                                st.caption(f"IPs: {', '.join(result['ip_addresses'])}")
                            else:
                                st.error(f"❌ {hostname}: {result['details']}")
                                if "error" in result:
                                    st.caption(f"Error: {result['error']}")
                        
                        # TCP Results
                        st.markdown("**🔌 TCP Connectivity:**")
                        for hostname, result in network_results["tcp_connectivity"].items():
                            if result["status"] == "success":
                                st.success(f"✅ {hostname}:443 - {result['details']}")
                            else:
                                st.error(f"❌ {hostname}:443 - {result['details']}")
                                if "error" in result:
                                    st.caption(f"Error: {result['error']}")
                        
                        # HTTP Results
                        st.markdown("**🌐 HTTP Connectivity:**")
                        for hostname, result in network_results["http_connectivity"].items():
                            if result["status"] == "success":
                                st.success(f"✅ {hostname} - {result['details']}")
                            else:
                                st.error(f"❌ {hostname} - {result['details']}")
                                if "error" in result:
                                    st.caption(f"Error: {result['error']}")
                        
                        # Recommendations
                        st.markdown("---")
                        st.subheader("💡 Recommendations")
                        
                        if failed > passed:
                            st.error("""
                            **Network connectivity issues detected:**
                            
                            1. **Check your internet connection** - Try accessing other websites
                            2. **Firewall/Proxy issues** - Your network may be blocking HTTPS connections to Datalab
                            3. **DNS issues** - Try using a different DNS server (8.8.8.8, 1.1.1.1)
                            4. **Try a different network** - Use mobile hotspot to test if it's network-specific
                            5. **Contact your IT administrator** if on a corporate network
                            
                            **For now, you can still use the Manual File Comparison feature with your own OCR outputs.**
                            """)
                        else:
                            st.info("Network connectivity looks good! If you're still experiencing issues, they may be temporary API server problems.")
                    
                    except Exception as e:
                        st.error(f"Diagnostics failed: {str(e)}")
        
        with col_diag2:
            if st.button("⚡ Quick Endpoint Test"):
                with st.spinner("Testing endpoints..."):
                    try:
                        diagnostic_results = _test_endpoints(settings)
                    except Exception as e:
                        st.error(f"Diagnostics failed: {e}")
                        diagnostic_results = []

                    if not diagnostic_results:
                        st.warning("No endpoints were tested. Check your configuration in config.toml.")
                    else:
                        for ep, status, info in diagnostic_results:
                            if status == "reachable":
                                st.success(f"✅ {ep} — reachable (HTTP {info})")
                            elif status in ("status",):
                                st.info(f"ℹ️ {ep} — responded with HTTP {info}")
                            else:
                                st.error(f"❌ {ep} — {status}: {info}")
        
        # Offline mode notice
        st.markdown("---")
        st.info("""
        **💡 Offline Mode Available:** If Datalab API is not accessible, you can still:
        1. Use the **Manual File Comparison** section to compare existing OCR outputs with ground truth
        2. Upload your own OCR-processed markdown files for evaluation
        3. Generate comprehensive evaluation reports and metrics
        """)
    
    # Side-by-side layout for file uploads
    col_pdf, col_gt = st.columns(2)
    
    with col_pdf:
        st.subheader("PDF File")
        uploaded_pdf = st.file_uploader(
            "Upload PDF file for OCR processing",
            type=["pdf"],
            key="auto_pdf_upload",
            help="Upload the PDF file to extract text from"
        )
        if uploaded_pdf:
            st.success(f"✅ PDF loaded: {uploaded_pdf.name}")
            with st.expander("Preview PDF"):
                _render_pdf(uploaded_pdf.getvalue(), height=400)
    
    with col_gt:
        st.subheader("Ground Truth File (Required)")
        uploaded_gt = st.file_uploader(
            "Upload Ground Truth (.md file)",
            type=["md"],
            key="auto_gt_upload",
            help="Upload the ground truth markdown file for comparison (required for evaluation)"
        )
        if uploaded_gt:
            st.success(f"✅ Ground truth loaded: {uploaded_gt.name}")
            with st.expander("Preview Ground Truth"):
                gt_content = uploaded_gt.getvalue().decode('utf-8')
                st.code(gt_content[:500] + "..." if len(gt_content) > 500 else gt_content, language="markdown")

    # Wrap Auto section for view toggling (already opened before header)
    # OCR Configuration Options
    if uploaded_pdf:
        st.subheader("OCR Configuration")
        
        col_model, col_lang = st.columns(2)
        
        with col_model:
            model_choice = st.radio(
                "Select OCR model",
                [
                    "Marker – Convert to Markdown/HTML/JSON",
                    "Surya – OCR text extraction",
                    "Deepseek – OCR via DeepInfra",
                    "Qwen3 VL 8B – via OpenRouter",
                ],
                key="auto_model_choice"
            )
        
        with col_lang:
            language_code = st.text_input(
                "Language Code (e.g., 'en', 'hi', 'ar')",
                value="",
                key="auto_language_code",
                help="Specify the language code for OCR to improve accuracy for regional languages. Leave blank for auto-detection."
            )
        
        # Determine model folder name
        if "Marker" in model_choice:
            model_folder = "Marker"
        elif "Surya" in model_choice:
            model_folder = "Surya"
        elif "Deepseek" in model_choice:
            model_folder = "Deepseek"
        else:
            model_folder = "Qwen3VL"
        
        # Model-specific options
        marker_options = {}
        ocr_options = {}
        deepseek_options: Dict[str, Any] = {}
        qwen_options: Dict[str, Any] = {}
        
        if "Marker" in model_choice:
            st.subheader("Marker Options")
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            
            # If a language code is provided, default to enabling LLM-enhanced processing and force OCR
            use_llm_default = bool(language_code)
            force_ocr_default = bool(language_code)

            with col_opt1:
                use_llm = st.checkbox(
                    "Use LLM-enhanced processing",
                    value=use_llm_default,
                    key="auto_use_llm",
                    help="Enable for complex layouts (higher latency/cost). Recommended for regional-language documents.",
                )
            
            with col_opt2:
                force_ocr = st.checkbox(
                    "Force OCR even if text is embedded",
                    value=force_ocr_default,
                    key="auto_force_ocr",
                    help="Force raster OCR rather than relying on embedded text. Recommended for scanned or regional-language PDFs.",
                )
            
            with col_opt3:
                max_pages_marker = st.number_input(
                    "Max pages to convert (0 = all pages)",
                    min_value=0,
                    value=0,
                    step=1,
                    key="auto_max_pages_marker",
                )
            
            marker_options = {
                "output_format": "markdown",
                "use_llm": use_llm,
                "force_ocr": force_ocr,
                "max_pages": max_pages_marker or None,
            }
            if language_code:
                marker_options["language"] = language_code
                st.info(
                    f"Language code '{language_code}' will be sent to the API. "
                    "Defaults: use_llm=True, force_ocr=True for better regional-language OCR."
                )
        elif "Surya" in model_choice:
            st.subheader("Surya OCR Options")
            max_pages_ocr = st.number_input(
                "Max pages to OCR (0 = all pages)",
                min_value=0,
                value=0,
                step=1,
                key="auto_max_pages_ocr",
            )
            ocr_options = {"max_pages": max_pages_ocr or None}
            if language_code:
                ocr_options["language"] = language_code
                st.info(
                    f"Language code '{language_code}' will be sent to the OCR endpoint to improve recognition."
                )
        elif "Deepseek" in model_choice:
            st.subheader("Deepseek OCR Options")
            col_ds1, col_ds2, col_ds3 = st.columns(3)
            with col_ds1:
                deepseek_mode = st.radio(
                    "Extraction mode",
                    ["markdown", "plain"],
                    index=0,
                    key="deepseek_mode",
                    help="Choose 'markdown' for document-like output or 'plain' for raw text."
                )
            with col_ds2:
                deepseek_max_pages = st.number_input(
                    "Max pages (0 = all)",
                    min_value=0,
                    value=0,
                    step=1,
                    key="deepseek_max_pages",
                )
            with col_ds3:
                deepseek_dpi = st.number_input(
                    "Render DPI",
                    min_value=72,
                    value=300,
                    step=24,
                    key="deepseek_dpi",
                    help="Higher DPI improves small text."
                )
            
            deepseek_adv_prompt = st.text_area(
                "Advanced prompt (optional)",
                value="",
                key="deepseek_adv_prompt",
                help="Provide custom instructions (e.g., 'Extract text as markdown.')."
            )
            deepseek_options = {
                "mode": deepseek_mode,
                "max_pages": deepseek_max_pages or None,
                "dpi": deepseek_dpi,
                "custom_prompt": (deepseek_adv_prompt or None),
            }
        else:
            st.subheader("Qwen3 VL Options")
            col_q1, col_q2, col_q3 = st.columns(3)
            with col_q1:
                qwen_mode = st.radio(
                    "Extraction mode",
                    ["markdown", "plain"],
                    index=0,
                    key="qwen_mode",
                    help="Choose 'markdown' for document-like output or 'plain' for raw text."
                )
            with col_q2:
                qwen_max_pages = st.number_input(
                    "Max pages (0 = all)",
                    min_value=0,
                    value=0,
                    step=1,
                    key="qwen_max_pages",
                )
            with col_q3:
                qwen_dpi = st.number_input(
                    "Render DPI",
                    min_value=72,
                    value=300,
                    step=24,
                    key="qwen_dpi",
                    help="Higher DPI improves small text."
                )
            col_q4, col_q5, col_q6 = st.columns(3)
            with col_q4:
                qwen_layout = st.selectbox(
                    "Layout",
                    options=["auto", "single", "two_column"],
                    index=0,
                    key="qwen_layout",
                    help="Reading order guidance. 'two_column' reads left column first, then right."
                )
            with col_q5:
                qwen_lang_hint = st.text_input(
                    "Language hint (e.g., te)",
                    value="",
                    key="qwen_language_hint",
                    help="Optional ISO code or note like 'Telugu'; prevents unintended translation."
                )
            with col_q6:
                qwen_enhance = st.checkbox(
                    "Enhance image",
                    value=False,
                    key="qwen_enhance_image",
                    help="Autocontrast and mild sharpening before OCR."
                )
            col_q7, col_q8, col_q9 = st.columns(3)
            with col_q7:
                qwen_tiling = st.checkbox(
                    "High-res tiling (2x2)",
                    value=False,
                    key="qwen_highres_tiling",
                    help="Split page into tiles to capture small text."
                )
            with col_q8:
                qwen_tile_rows = st.number_input(
                    "Tile rows",
                    min_value=1,
                    value=2,
                    step=1,
                    key="qwen_tile_rows",
                    disabled=not st.session_state.get("qwen_highres_tiling", False)
                )
            with col_q9:
                qwen_tile_cols = st.number_input(
                    "Tile cols",
                    min_value=1,
                    value=2,
                    step=1,
                    key="qwen_tile_cols",
                    disabled=not st.session_state.get("qwen_highres_tiling", False)
                )

            col_q10, col_q11 = st.columns(2)
            with col_q10:
                qwen_temperature = st.number_input(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1,
                    format="%.1f",
                    key="qwen_temperature",
                    help="Lower (0.0) reduces hallucinations."
                )
            with col_q11:
                qwen_single_req = st.checkbox(
                    "Combine tiles in one request",
                    value=True,
                    key="qwen_multi_image_single_request",
                    help="Uncheck to send each tile separately for stricter ordering."
                )

            qwen_adv_prompt = st.text_area(
                "Advanced prompt (optional)",
                value="",
                key="qwen_adv_prompt",
                help="Provide custom instructions (e.g., 'Extract text as markdown.')."
            )
            qwen_options = {
                "mode": qwen_mode,
                "max_pages": qwen_max_pages or None,
                "dpi": qwen_dpi,
                "custom_prompt": (qwen_adv_prompt or None),
                "layout": qwen_layout,
                "language_hint": (qwen_lang_hint or None),
                "enhance_image": qwen_enhance,
                "highres_tiling": qwen_tiling,
                "tile_rows": qwen_tile_rows if qwen_tiling else 1,
                "tile_cols": qwen_tile_cols if qwen_tiling else 1,
                "temperature_override": qwen_temperature,
                "multi_image_single_request": qwen_single_req,
            }

        # Process button
        st.subheader("Processing")
        col_btn, col_eval, col_multirun = st.columns([1, 2, 2])
        
        with col_eval:
            # Run mode selection: mutually exclusive radio between single-run (auto-evaluate) and multi-run
            run_mode = st.radio(
                "Run Mode",
                ["Single Run (Auto-evaluate)", "Multi-Run Consistency Evaluation"],
                index=1 if settings.multi_run_enabled else 0,
                key="auto_run_mode",
                help="Choose Single Run to run once and auto-evaluate, or Multi-Run to perform multiple independent runs for consistency analysis."
            )
            if run_mode.startswith("Single Run"):
                st.checkbox(
                    "Ignore image descriptions (captions/alt text)",
                    value=False,
                    key="single_ignore_img_desc",
                    help="When enabled, strips Markdown/HTML image tags and common caption lines (Figure/Caption/etc.) from both GT and OCR before computing metrics."
                )
        
        with col_multirun:
            if run_mode.startswith("Multi-Run"):
                st.info("🔄 Multi-Run selected: Will perform multiple independent runs for consistency analysis (default 3 runs). A 5 second delay will be applied before each run.")
        
        with col_btn:
            # Single consistent 'Run' button per request
            run_auto_extraction = st.button("🚀 Run", type="primary", key="auto_run_btn", disabled=not uploaded_gt)
            
            if not uploaded_gt:
                st.warning("⚠️ Ground truth file is required for evaluation")

    def process_single_file(uploaded_file, model_choice, model_folder, marker_options, ocr_options, language_code, settings, uploaded_gt_file=None, enable_comparison=True):
        """Process a single PDF file and return results"""
        file_bytes = uploaded_file.getvalue()
        results = {
            'filename': uploaded_file.name,
            'success': False,
            'markdown': None,
            'gt_path': None,
            'ocr_path': None,
            'eval_summary': None,
            'eval_chart': None,
            'error': None
        }
        
        try:
            # Submit job to API / local DeepSeek
            if "Marker" in model_choice:
                result_payload = _submit_marker_job(
                    file_bytes=file_bytes,
                    filename=uploaded_file.name,
                    mime_type=uploaded_file.type or "application/pdf",
                    settings=settings,
                    **marker_options,
                )
            elif "Surya" in model_choice:
                result_payload = _submit_ocr_job(
                    file_bytes=file_bytes,
                    filename=uploaded_file.name,
                    mime_type=uploaded_file.type or "application/pdf",
                    settings=settings,
                    **ocr_options,
                )
            elif "Deepseek" in model_choice:
                result_payload = _submit_deepseek_job(
                    file_bytes=file_bytes,
                    filename=uploaded_file.name,
                    mime_type=uploaded_file.type or "application/pdf",
                    settings=get_deepinfra_settings(),
                    **deepseek_options,
                )
            else:
                result_payload = _submit_qwen_job(
                    file_bytes=file_bytes,
                    filename=uploaded_file.name,
                    mime_type=uploaded_file.type or "application/pdf",
                    settings=get_openrouter_settings(),
                    **qwen_options,
                )
            
            markdown = _extract_markdown(result_payload)
            results['markdown'] = markdown
            
            # Save OCR output to model-specific folder and write model metadata separately
            if markdown:
                model_output_dir = os.path.join("OCR_Output", model_folder)
                os.makedirs(model_output_dir, exist_ok=True)
                pdf_name = os.path.splitext(uploaded_file.name)[0]
                
                # Save OCR markdown to model-specific folder (no metadata appended)
                md_file_path = os.path.join(model_output_dir, f"{pdf_name}_ocr_output.md")
                with open(md_file_path, 'w', encoding='utf-8') as f:
                    f.write(markdown)
                results['ocr_path'] = md_file_path
                
                # Build model metadata (exclude output-specific metrics like word/char counts)
                try:
                    ts = datetime.now(timezone.utc).astimezone()
                    timestamp = ts.isoformat()
                    tz_name = ts.tzname() or time.tzname[0]
                except Exception:
                    timestamp = datetime.now().isoformat()
                    tz_name = time.tzname[0] if hasattr(time, "tzname") else "UTC"
                
                metadata = {
                    "model_choice": model_choice,
                    "model_folder": model_folder,
                    "language_code": language_code or "",
                    "timestamp": timestamp,
                    "timezone": tz_name,
                }
                
                # Include model-specific settings only
                try:
                    if marker_options:
                        metadata["marker_options"] = marker_options
                except Exception:
                    pass
                try:
                    if ocr_options:
                        metadata["ocr_options"] = ocr_options
                except Exception:
                    pass
                try:
                    if deepseek_options:
                        metadata["deepseek_options"] = deepseek_options
                    if 'qwen_options' in locals() and qwen_options:
                        metadata["qwen_options"] = qwen_options
                except Exception:
                    pass
                
                # Write metadata to a JSON file next to the OCR output
                metadata_path = os.path.join(model_output_dir, f"{pdf_name}_metadata.json")
                try:
                    with open(metadata_path, 'w', encoding='utf-8') as mf:
                        json.dump(metadata, mf, ensure_ascii=False, indent=2)
                except Exception:
                    # Fall back to writing a simple text file if JSON dump fails
                    try:
                        with open(metadata_path, 'w', encoding='utf-8') as mf:
                            mf.write(str(metadata))
                    except Exception:
                        pass
                
                # Run evaluation if ground truth file is provided and comparison is enabled
                if enable_comparison and uploaded_gt_file and markdown:
                    try:
                        from compare import evaluate_ocr_performance
                        
                        # Use gt_text parameter to pass uploaded GT content directly (no temp file needed)
                        gt_text = uploaded_gt_file.getvalue().decode('utf-8')
                        
                        summary_txt_path, summary_md_path, chart_path = evaluate_ocr_performance(
                            gt_text=gt_text,
                            ocr_file=md_file_path,
                            output_dir=os.path.join("evaluation/results", model_folder),
                            display_model_name=model_choice,
                            gt_display_name=f"{uploaded_gt_file.name} (Ground Truth)",
                            ocr_display_name=f"{uploaded_file.name} (OCR Output)"
                        )
                        results['eval_summary'] = summary_txt_path
                        results['eval_summary_md'] = summary_md_path
                        results['eval_chart'] = chart_path
                        
                        # Store additional metrics for consistent display with multi-run
                        results['metrics_computed'] = True
                    except Exception as eval_error:
                        try:
                            import traceback
                            tb = traceback.format_exc()
                        except Exception:
                            tb = None
                        results['error'] = f"Evaluation failed: {type(eval_error).__name__}: {str(eval_error)}"
                        if tb:
                            results['error_trace'] = tb
                        results['metrics_computed'] = False
            
            results['success'] = True
            
        except Exception as exc:
            results['error'] = str(exc)
        
        return results

    # Auto processing logic - moved outside the function definition
    if uploaded_pdf and uploaded_gt and run_auto_extraction:
        if run_mode.startswith("Multi-Run"):
            # Multi-run evaluation
            with st.spinner("Initializing multi-run evaluation..."):
                try:
                    # Initialize multi-run components
                    runner = MultiRunRunner(num_runs=settings.multi_run_runs)
                    reporter = MultiRunReporter()
                    
                    # Get PDF basename for directory structure
                    pdf_basename = get_pdf_basename(uploaded_pdf.name)
                    
                    # Generate ground truth first (prefer user-uploaded ground truth if provided)
                    file_bytes = uploaded_pdf.getvalue()
                    
                    # Use the uploaded ground truth file for multi-run evaluation
                    try:
                        gt_text = uploaded_gt.getvalue().decode('utf-8')
                        if not gt_text.strip():
                            st.error("Ground truth file is empty")
                            st.stop()
                    except Exception as e:
                        st.error(f"Failed to read ground truth file: {str(e)}")
                        st.stop()
                    
                    # Create extraction wrapper function
                    if "Marker" in model_choice:
                        extraction_func = create_extraction_wrapper(_submit_marker_job, _extract_markdown)
                        extraction_params = {
                            'file_bytes': file_bytes,
                            'filename': uploaded_pdf.name,
                            'mime_type': uploaded_pdf.type or "application/pdf",
                            'settings': settings,
                            'model_name': 'Marker',
                            'model_version': 'v1.0',
                            'model_family': 'LLM',
                            **marker_options
                        }
                    elif "Surya" in model_choice:
                        extraction_func = create_extraction_wrapper(_submit_ocr_job, _extract_markdown)
                        extraction_params = {
                            'file_bytes': file_bytes,
                            'filename': uploaded_pdf.name,
                            'mime_type': uploaded_pdf.type or "application/pdf",
                            'settings': settings,
                            'model_name': 'Surya',
                            'model_version': 'v1.0',
                            'model_family': 'OCR',
                            **ocr_options
                        }
                    elif "Deepseek" in model_choice:
                        extraction_func = create_extraction_wrapper(_submit_deepseek_job, _extract_markdown)
                        extraction_params = {
                            'file_bytes': file_bytes,
                            'filename': uploaded_pdf.name,
                            'mime_type': uploaded_pdf.type or "application/pdf",
                            'model_name': 'Deepseek',
                            'model_version': 'v1.0',
                            'model_family': 'OCR',
                            **deepseek_options
                        }
                    else:
                        extraction_func = create_extraction_wrapper(_submit_qwen_job, _extract_markdown)
                        extraction_params = {
                            'file_bytes': file_bytes,
                            'filename': uploaded_pdf.name,
                            'mime_type': uploaded_pdf.type or "application/pdf",
                            'model_name': 'Qwen3 VL 8B',
                            'model_version': 'v1.0',
                            'model_family': 'OCR',
                            **qwen_options
                        }
                    
                    # If multi-run mode selected, wrap extraction_func to add a 5 second delay before each run
                    try:
                        if 'run_mode' in locals() and run_mode.startswith("Multi-Run"):
                            original_extraction = extraction_func
                            def extraction_with_delay(**params):
                                # Delay before each run to allow deterministic spacing between runs
                                time.sleep(5)
                                return original_extraction(**params)
                            extraction_func = extraction_with_delay
                    except Exception:
                        # If anything goes wrong, fall back to the original extraction_func
                        pass
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def progress_callback(message):
                        status_text.text(message)
                        # Update progress based on message content
                        if "Run 1:" in message:
                            progress_bar.progress(0.1)
                        elif "Run 2:" in message:
                            progress_bar.progress(0.4)
                        elif "Run 3:" in message:
                            progress_bar.progress(0.7)
                        elif "Computing aggregated" in message:
                            progress_bar.progress(0.9)
                        elif "completed" in message:
                            progress_bar.progress(1.0)
                    
                    # Execute multi-run evaluation
                    summary = runner.execute_multi_run_evaluation(
                        pdf_basename=pdf_basename,
                        model_folder=model_folder,
                        gt_text=gt_text,
                        extraction_func=extraction_func,
                        extraction_params=extraction_params,
                        progress_callback=progress_callback
                    )
                    
                    # Generate comprehensive report
                    status_text.text("Generating comprehensive report...")
                    
                    # Load run metrics for report generation
                    run_metrics_list = []
                    for run_detail in summary.run_details:
                        if run_detail['status'] == 'success':
                            metrics_path = os.path.join(
                                runner.output_base_dir, model_folder, pdf_basename,
                                run_detail['metrics_file']
                            )
                            if os.path.exists(metrics_path):
                                with open(metrics_path, 'r', encoding='utf-8') as f:
                                    metrics_data = json.load(f)
                                
                                from multi_run_evaluation import RunMetrics
                                # The metrics_data is already the RunMetrics structure
                                # Ensure proper type conversion from JSON
                                run_metrics = RunMetrics(
                                    run_id=int(metrics_data['run_id']),
                                    wer=float(metrics_data['wer']),
                                    mer=float(metrics_data['mer']),
                                    wil=float(metrics_data['wil']),
                                    cer=float(metrics_data['cer']),
                                    lev_distance=int(metrics_data['lev_distance']),
                                    lev_norm=float(metrics_data['lev_norm']),
                                    structural_accuracy=metrics_data['structural_accuracy'],
                                    structural_analysis=metrics_data.get('structural_analysis', {}),
                                    completeness=float(metrics_data['completeness']),
                                    word_mismatches=metrics_data['word_mismatches']
                                )
                                run_metrics_list.append(run_metrics)
                    
                    report_md_path, report_txt_path, chart_paths = reporter.generate_comprehensive_report(
                        summary, run_metrics_list, model_folder
                    )
                    
                    # After report generation, ensure each run's raw output is also saved with the PDF basename
                    try:
                        for run_detail in summary.run_details:
                            run_id = run_detail.get("run_id")
                            run_raw_path = os.path.join(runner.output_base_dir, model_folder, pdf_basename, f"run_{run_id}", "raw_output.md")
                            if os.path.exists(run_raw_path):
                                dest_path = os.path.join(runner.output_base_dir, model_folder, pdf_basename, f"run_{run_id}", f"{pdf_basename}.md")
                                try:
                                    shutil.copyfile(run_raw_path, dest_path)
                                except Exception:
                                    # Non-fatal; continue
                                    pass
                    except Exception:
                        pass
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Multi-run evaluation completed!")
                    
                    # Display results
                    st.success(f"🎉 Multi-run evaluation completed successfully!")
                    
                    # Summary metrics
                    col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
                    with col_summary1:
                        st.metric("Total Runs", summary.total_runs)
                    with col_summary2:
                        st.metric("Successful Runs", summary.successful_runs)
                    with col_summary3:
                        st.metric("Overall CCI", f"{summary.overall_cci:.4f}")
                    with col_summary4:
                        stability_color = "normal" if summary.overall_cci > 0.90 else "inverse" if summary.overall_cci >= 0.75 else "off"
                        st.metric("Stability", summary.stability_interpretation)
                    
                    # Top-level Final Report tab for visibility
                    try:
                        consolidated_lines = []
                        consolidated_lines.append(f"# 🧾 Final Consolidated Report\n")
                        consolidated_lines.append(f"**PDF:** `{pdf_basename}`  ")
                        consolidated_lines.append(f"**Model:** `{model_folder}`  ")
                        consolidated_lines.append(f"**Total Runs:** {summary.total_runs}  ")
                        consolidated_lines.append(f"**Successful Runs:** {summary.successful_runs}  ")
                        consolidated_lines.append(f"**Overall CCI:** {summary.overall_cci:.4f}  ")
                        consolidated_lines.append(f"**Stability:** {summary.stability_interpretation}  \n")

                        # Aggregate Scores table
                        try:
                            consolidated_lines.append("## Aggregate Scores\n")
                            consolidated_lines.append("| Metric | Score |")
                            consolidated_lines.append("|--------|-------|")
                            consolidated_lines.append(f"| **Text Accuracy Score** | {((f'{summary.text_score:.4f}') if summary.text_score is not None else 'NA')} |")
                            consolidated_lines.append(f"| **Structural Score** | {((f'{summary.structural_score:.4f}') if summary.structural_score is not None else 'NA')} |")
                            consolidated_lines.append(f"| **Overall Extraction Score** | {((f'{summary.overall_score:.4f}') if summary.overall_score is not None else 'NA')} |\n")
                        except Exception:
                            pass

                        consolidated_lines.append("## Per-Run Metrics\n")
                        consolidated_lines.append(reporter.generate_per_run_metrics_table(run_metrics_list))
                        consolidated_lines.append("## Aggregated Metrics\n")
                        consolidated_lines.append(reporter.generate_aggregated_metrics_table(summary.aggregated_metrics))

                        consolidated_md_top = "\n".join(consolidated_lines)
                        top_final_tab = st.tabs(["Final Report"])[0]
                        with top_final_tab:
                            st.markdown(consolidated_md_top)
                            # Info expander with consolidated ideal ranges table
                            with st.expander("ℹ️ Ideal Evaluation Metric Ranges (Consolidated Table)"):
                                st.markdown(_ideal_ranges_table_md())
                    except Exception as e:
                        st.warning(f"Could not render top-level Final Report: {e}")

                    # Display comprehensive report (organized)
                    with st.expander("📊 Multi-Run Evaluation Report", expanded=True):
                        tab_final, tab_summary, tab_charts, tab_comparisons = st.tabs(["Final Report", "Full Report", "Charts", "Comparisons"])

                        # Final Report tab (Consolidated)
                        with tab_final:
                            try:
                                consolidated_lines = []
                                consolidated_lines.append(f"# 🧾 Final Consolidated Report\n")
                                consolidated_lines.append(f"**PDF:** `{pdf_basename}`  ")
                                consolidated_lines.append(f"**Model:** `{model_folder}`  ")
                                consolidated_lines.append(f"**Total Runs:** {summary.total_runs}  ")
                                consolidated_lines.append(f"**Successful Runs:** {summary.successful_runs}  ")
                                consolidated_lines.append(f"**Overall CCI:** {summary.overall_cci:.4f}  ")
                                consolidated_lines.append(f"**Stability:** {summary.stability_interpretation}  \n")

                                # Aggregate Scores
                                try:
                                    consolidated_lines.append("## Aggregate Scores\n")
                                    consolidated_lines.append("| Metric | Score |")
                                    consolidated_lines.append("|--------|-------|")
                                    consolidated_lines.append(f"| **Text Accuracy Score** | {((f'{summary.text_score:.4f}') if summary.text_score is not None else 'NA')} |")
                                    consolidated_lines.append(f"| **Structural Score** | {((f'{summary.structural_score:.4f}') if summary.structural_score is not None else 'NA')} |")
                                    consolidated_lines.append(f"| **Overall Extraction Score** | {((f'{summary.overall_score:.4f}') if summary.overall_score is not None else 'NA')} |\n")
                                except Exception:
                                    pass

                                # Per-run metrics table
                                consolidated_lines.append("## Per-Run Metrics\n")
                                consolidated_lines.append(reporter.generate_per_run_metrics_table(run_metrics_list))

                                # Aggregated metrics table
                                consolidated_lines.append("## Aggregated Metrics\n")
                                consolidated_lines.append(reporter.generate_aggregated_metrics_table(summary.aggregated_metrics))

                                consolidated_md = "\n".join(consolidated_lines)
                                st.markdown(consolidated_md)
                                # Info expander with consolidated ideal ranges table
                                with st.expander("ℹ️ Ideal Evaluation Metric Ranges (Consolidated Table)"):
                                    st.markdown(_ideal_ranges_table_md())
                            except Exception as e:
                                st.warning(f"Could not render Final Report: {e}")

                        # Full Report tab
                        with tab_summary:
                            if os.path.exists(report_md_path):
                                try:
                                    with open(report_md_path, 'r', encoding='utf-8') as f:
                                        report_content = f.read()
                                    st.markdown(report_content)
                                except Exception as e:
                                    st.warning(f"Could not read report: {e}")
                            else:
                                st.error("Multi-run report not available")

                        # Charts tab
                        with tab_charts:
                            if chart_paths:
                                for chart_path in chart_paths:
                                    if os.path.exists(chart_path):
                                        chart_name = os.path.basename(chart_path)
                                        if "variance" in chart_name:
                                            st.markdown("**📊 Metrics Variance Chart**")
                                        elif "comparison" in chart_name:
                                            st.markdown("**🔄 Run Comparison Chart**")
                                        st.image(chart_path, use_container_width=True)
                                    else:
                                        st.warning(f"Chart file not found: {chart_path}")
                            else:
                                st.info("No charts generated.")

                        # Comparisons tab
                        with tab_comparisons:
                            try:
                                st.subheader("📋 Side-by-side File Comparisons (per run)")
                                for run_detail in summary.run_details:
                                    if run_detail.get("status") != "success":
                                        continue
                                    run_id = run_detail.get("run_id")
                                    run_raw_path = os.path.join(runner.output_base_dir, model_folder, pdf_basename, f"run_{run_id}", "raw_output.md")
                                    if not os.path.exists(run_raw_path):
                                        st.warning(f"Raw output for run {run_id} not found: {run_raw_path}")
                                        continue
                                    with open(run_raw_path, 'r', encoding='utf-8') as rf:
                                        ocr_content = rf.read()

                                    # Ground truth used for multi-run (prefer uploaded GT if present)
                                    if uploaded_gt:
                                        gt_display_text = uploaded_gt.getvalue().decode('utf-8')
                                    else:
                                        # gt_text was created earlier in this flow
                                        gt_display_text = gt_text if 'gt_text' in locals() else ""

                                    st.markdown(f"#### Run {run_id}")
                                    try:
                                        diff_html = render_side_by_side_diff(gt_display_text, ocr_content, gt_name="Ground Truth", ocr_name=f"Run {run_id} OCR Output")
                                        st.markdown(diff_html, unsafe_allow_html=True)
                                    except Exception as e:
                                        st.warning(f"Could not render diff for run {run_id}: {e}")
                            except Exception as multiex:
                                st.warning(f"Side-by-side comparisons unavailable: {multiex}")
                    
                    # Download section
                    st.subheader("💾 Downloads")
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    
                    with col_dl1:
                        if os.path.exists(report_md_path):
                            with open(report_md_path, 'r', encoding='utf-8') as f:
                                md_content = f.read()
                            st.download_button(
                                label="📄 Download Report (Markdown)",
                                data=md_content,
                                file_name=f"multirun_report_{pdf_basename}.md",
                                mime="text/markdown"
                            )
                    
                    with col_dl2:
                        if os.path.exists(report_txt_path):
                            with open(report_txt_path, 'r', encoding='utf-8') as f:
                                txt_content = f.read()
                            st.download_button(
                                label="📝 Download Report (Text)",
                                data=txt_content,
                                file_name=f"multirun_report_{pdf_basename}.txt",
                                mime="text/plain"
                            )
                    
                    with col_dl3:
                        # Create summary JSON for download
                        from dataclasses import asdict
                        summary_dict = asdict(summary)
                        summary_dict['aggregated_metrics'] = {
                            k: asdict(v) for k, v in summary.aggregated_metrics.items()
                        }
                        summary_json = json.dumps(summary_dict, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="📊 Download Summary (JSON)",
                            data=summary_json,
                            file_name=f"multirun_summary_{pdf_basename}.json",
                            mime="application/json"
                        )
                    
                except Exception as e:
                    st.error(f"Multi-run evaluation failed: {str(e)}")
                    st.exception(e)
        else:
            # Single run evaluation (existing logic)
            with st.spinner("Processing PDF with OCR..."):
                results = process_single_file(
                    uploaded_pdf, model_choice, model_folder,
                    marker_options, ocr_options, language_code, settings, uploaded_gt, True  # Always enable comparison for single run
                )

                if results['success']:
                    # Display extracted text in a collapsible expander
                    with st.expander("📄 Extracted Text Output", expanded=True):
                        if results['markdown']:
                            st.code(results['markdown'], language="markdown")
                            st.success(f"✅ OCR extraction completed successfully!")
                            if results['ocr_path']:
                                st.info(f"📝 OCR output saved to: {results['ocr_path']}")
                        else:
                            st.warning("No textual output was returned by the API.")

                    # If ground truth was provided, run comprehensive evaluation
                    if uploaded_gt and results.get('ocr_path'):
                        try:
                            import tempfile
                            from compare import evaluate_ocr_performance

                            def generate_report_name(gt_filename, pdf_filename):
                                gt_base = os.path.splitext(gt_filename)[0]
                                pdf_base = os.path.splitext(pdf_filename)[0]
                                gt_short = gt_base[:6] if len(gt_base) > 6 else gt_base
                                pdf_short = pdf_base[:6] if len(pdf_base) > 6 else pdf_base
                                return f"{gt_short}_{pdf_short}"

                            custom_report_name = generate_report_name(uploaded_gt.name, uploaded_pdf.name)

                            # Use gt_text parameter to pass uploaded GT content directly (no temp file needed)
                            gt_text = uploaded_gt.getvalue().decode('utf-8')
                            
                            out_dir = os.path.join("evaluation/results", model_folder)
                            os.makedirs(out_dir, exist_ok=True)

                            # Read Single Run option from earlier control
                            ignore_image_desc = st.session_state.get("single_ignore_img_desc", False)

                            summary_txt_path, summary_md_path, chart_path = evaluate_ocr_performance(
                                gt_text=gt_text,
                                ocr_file=results['ocr_path'],
                                output_dir=out_dir,
                                custom_name=custom_report_name,
                                display_model_name=model_choice,
                                gt_display_name=uploaded_gt.name,
                                ocr_display_name=uploaded_pdf.name,
                                ignore_image_desc=ignore_image_desc
                            )
    
                            # Display comprehensive evaluation report (organized)
                            with st.expander("📊 Comprehensive Evaluation Report", expanded=True):
                                st.markdown(f"**📋 {uploaded_pdf.name}** vs **{uploaded_gt.name}** • Model: {model_choice.split(' –')[0]} • {datetime.now().strftime('%H:%M')}")
                                st.markdown("---")

                                tab_overview, tab_chart, tab_report, tab_diff = st.tabs(["Overview", "Metrics Chart", "Full Report", "Side-by-Side"])

                                # Overview tab: quick info + downloads
                                with tab_overview:
                                    cols = st.columns(3)
                                    with cols[0]:
                                        st.metric("Ground Truth", uploaded_gt.name)
                                    with cols[1]:
                                        st.metric("PDF", uploaded_pdf.name)
                                    with cols[2]:
                                        st.metric("Model", model_choice.split(' –')[0])

                                    if st.session_state.get("single_ignore_img_desc", False):
                                        st.caption("Preprocessing: image descriptions ignored")

                                    dl_cols = st.columns(3)
                                    # Report (Markdown)
                                    if os.path.exists(summary_md_path):
                                        try:
                                            with open(summary_md_path, 'r', encoding='utf-8') as f:
                                                md_data = f.read()
                                            with dl_cols[0]:
                                                st.download_button("⬇️ Report (Markdown)", data=md_data, file_name=os.path.basename(summary_md_path), mime="text/markdown")
                                        except Exception:
                                            pass
                                    # Report (Text)
                                    if os.path.exists(summary_txt_path):
                                        try:
                                            with open(summary_txt_path, 'r', encoding='utf-8') as f:
                                                txt_data = f.read()
                                            with dl_cols[1]:
                                                st.download_button("⬇️ Report (Text)", data=txt_data, file_name=os.path.basename(summary_txt_path), mime="text/plain")
                                        except Exception:
                                            pass
                                    # Chart (PNG)
                                    if os.path.exists(chart_path):
                                        try:
                                            with open(chart_path, 'rb') as f:
                                                png_bytes = f.read()
                                            with dl_cols[2]:
                                                st.download_button("⬇️ Chart (PNG)", data=png_bytes, file_name=os.path.basename(chart_path), mime="image/png")
                                        except Exception:
                                            pass

                                # Metrics Chart tab
                                with tab_chart:
                                    if os.path.exists(chart_path):
                                        st.image(chart_path, use_container_width=True)
                                    else:
                                        st.warning("Chart not available")

                                # Full Report tab
                                with tab_report:
                                    if os.path.exists(summary_md_path):
                                        try:
                                            with open(summary_md_path, 'r', encoding='utf-8') as f:
                                                md_content = f.read()
                                            st.markdown(md_content)
                                        except Exception as e:
                                            st.warning(f"Could not read report: {e}")
                                    else:
                                        st.error("Report not available")

                                # Side-by-Side tab
                                with tab_diff:
                                    try:
                                        # Optional: ignore image descriptions in side-by-side diff
                                        ignore_img_desc_diff = st.checkbox(
                                            "Ignore image descriptions in diff",
                                            value=st.session_state.get("single_ignore_img_desc", False),
                                            key="single_ignore_img_desc_diff",
                                            help="Applies the same image/caption stripping when rendering the side-by-side view (Single Run only)."
                                        )
                                        # Determine ground truth content used for this evaluation
                                        if uploaded_gt:
                                            gt_content = uploaded_gt.getvalue().decode('utf-8')
                                        elif results.get('gt_path') and os.path.exists(results['gt_path']):
                                            with open(results['gt_path'], 'r', encoding='utf-8') as gf:
                                                gt_content = gf.read()
                                        else:
                                            gt_content = ""

                                        if gt_content and results.get('ocr_path') and os.path.exists(results['ocr_path']):
                                            with open(results['ocr_path'], 'r', encoding='utf-8') as of:
                                                ocr_content = of.read()

                                            if ignore_img_desc_diff:
                                                gt_content = _strip_image_descriptions(gt_content)
                                                ocr_content = _strip_image_descriptions(ocr_content)

                                            try:
                                                diff_html = render_side_by_side_diff(gt_content, ocr_content, gt_name="Ground Truth", ocr_name=uploaded_pdf.name)
                                                st.markdown(diff_html, unsafe_allow_html=True)
                                            except Exception as e:
                                                st.warning(f"Could not render side-by-side diff: {e}")
                                        else:
                                            st.info("Insufficient data to render comparison.")
                                    except Exception as ex:
                                        st.warning(f"Side-by-side comparison unavailable: {ex}")


                        except Exception as e:
                            try:
                                import traceback
                                tb = traceback.format_exc()
                            except Exception:
                                tb = None
                            st.error(f"Evaluation failed: {type(e).__name__}: {str(e)}")
                            if tb:
                                with st.expander("Show error details"):
                                    st.code(tb)


                else:
                    st.error(f"Processing failed: {results['error']}")
                    
                    # Provide specific guidance for network connectivity issues
                    if "All configured Datalab endpoints failed" in str(results.get('error', '')):
                        st.markdown("""
                        ### 🔧 Network Connectivity Issue Detected
                        
                        The Datalab API servers are not reachable from your network. This could be due to:
                        
                        **Common Causes:**
                        - **Internet connectivity issues** - Check your internet connection
                        - **Firewall/Proxy blocking** - Your network may block HTTPS connections to external APIs
                        - **DNS resolution problems** - Try using different DNS servers (8.8.8.8, 1.1.1.1)
                        - **Corporate network restrictions** - Contact your IT administrator
                        - **Temporary API server issues** - The Datalab servers may be temporarily unavailable
                        
                        **What you can do:**
                        1. **Run Network Diagnostics** - Use the "🔍 Run Full Network Diagnostics" button above to identify the specific issue
                        2. **Try a different network** - Use mobile hotspot to test if it's network-specific
                        3. **Use Manual File Comparison** - You can still evaluate OCR outputs you have from other sources
                        4. **Try again later** - API connectivity issues are often temporary
                        """)
                        
                        # Show quick diagnostic button
                        if st.button("🔍 Run Quick Network Test", key="error_diagnostic"):
                            with st.spinner("Testing network connectivity..."):
                                try:
                                    network_results = _test_network_connectivity()
                                    failed = network_results["summary"]["failed"]
                                    passed = network_results["summary"]["passed"]
                                    
                                    if failed > passed:
                                        st.error("❌ Network connectivity issues confirmed. See diagnostics section above for detailed analysis.")
                                    else:
                                        st.success("✅ Network connectivity appears normal. This may be a temporary API server issue.")
                                except Exception as e:
                                    st.error(f"Network test failed: {str(e)}")
                    else:
                        st.warning(f"⚠️ {results['error']}")


def render_batch_section(settings):
    """Render the Batch Multi-Run section"""
    # End of Batch section
    # ========================================
    # BATCH: AUTO-MAP GT BY PDF NAME (MULTI-RUN)
    # ========================================
    st.header("📂 Batch: Auto‑Map Ground Truth (Multi‑Run)")
    st.caption("Upload one or more PDFs; we map to ground‑truth Markdown by filename and run a multi‑run evaluation per file.")

    def _find_ground_truth_for_pdf(pdf_filename: str) -> Optional[str]:
        """Find a ground truth Markdown file by matching PDF base name across ground_truth/ recursively.
        Matches candidates: <base>.md, <base>_ground_truth.md (case-insensitive). Returns first match path or None.
        """
        base = get_pdf_basename(pdf_filename)
        candidates = [f"{base}.md", f"{base}_ground_truth.md"]
        gt_root = os.path.join("ground_truth")
        for root, _, files in os.walk(gt_root):
            lower_map = {f.lower(): f for f in files}
            for cand in candidates:
                lc = cand.lower()
                if lc in lower_map:
                    return os.path.join(root, lower_map[lc])
        return None

    col_batch_left, col_batch_right = st.columns([2, 2])
    with col_batch_left:
        batch_pdfs = st.file_uploader(
            "Upload PDF file(s)",
            type=["pdf"],
            accept_multiple_files=True,
            key="batch_pdf_upload",
            help="Upload one or more PDFs. We'll search ground_truth/ for a matching .md by filename."
        )

    with col_batch_right:
        model_choice_batch = st.radio(
            "Select model for extraction",
            [
                "Marker – Convert to Markdown/HTML/JSON",
                "Surya – OCR text extraction",
                "Deepseek – OCR via DeepInfra",
                "Qwen3 VL 8B – via OpenRouter",
            ],
            key="batch_model_choice"
        )
        language_code_batch = st.text_input(
            "Language Code (optional)",
            value="",
            key="batch_language_code",
            help="e.g., en, hi, ar. Improves OCR for regional languages."
        )

    marker_options_batch = {}
    ocr_options_batch = {}
    deepseek_options_batch = {}
    qwen_options_batch = {}
    if "Marker" in model_choice_batch:
        model_folder_batch = "Marker"
    elif "Surya" in model_choice_batch:
        model_folder_batch = "Surya"
    elif "Deepseek" in model_choice_batch:
        model_folder_batch = "Deepseek"
    else:
        model_folder_batch = "Qwen3VL"

    if "Marker" in model_choice_batch:
        st.subheader("Marker Options (Batch)")
        cb1, cb2, cb3 = st.columns(3)
        use_llm_default_b = bool(language_code_batch)
        force_ocr_default_b = bool(language_code_batch)
        with cb1:
            use_llm_b = st.checkbox("Use LLM", value=use_llm_default_b, key="batch_use_llm")
        with cb2:
            force_ocr_b = st.checkbox("Force OCR", value=force_ocr_default_b, key="batch_force_ocr")
        with cb3:
            max_pages_b = st.number_input("Max pages (0=all)", min_value=0, value=0, step=1, key="batch_max_pages_marker")
        marker_options_batch = {"output_format": "markdown", "use_llm": use_llm_b, "force_ocr": force_ocr_b, "max_pages": max_pages_b or None}
        if language_code_batch:
            marker_options_batch["language"] = language_code_batch
    elif "Surya" in model_choice_batch:
        st.subheader("Surya Options (Batch)")
        max_pages_ocr_b = st.number_input("Max pages", min_value=0, value=0, step=1, key="batch_max_pages_ocr")
        ocr_options_batch = {"max_pages": max_pages_ocr_b or None}
        if language_code_batch:
            ocr_options_batch["language"] = language_code_batch
    elif "Deepseek" in model_choice_batch: # Deepseek
        st.subheader("Deepseek Options (Batch)")
        col_b_ds1, col_b_ds2, col_b_ds3 = st.columns(3)
        with col_b_ds1:
            deepseek_mode_b = st.radio("Extraction mode", ["markdown", "plain"], index=0, key="batch_deepseek_mode")
        with col_b_ds2:
            deepseek_max_pages_b = st.number_input("Max pages", min_value=0, value=0, step=1, key="batch_deepseek_max_pages")
        with col_b_ds3:
            deepseek_dpi_b = st.number_input("Render DPI", min_value=72, value=300, step=24, key="batch_deepseek_dpi")
        
        deepseek_adv_prompt_b = st.text_area("Advanced prompt (optional)", value="", key="batch_deepseek_adv_prompt")
        
        deepseek_options_batch = {
            "mode": deepseek_mode_b,
            "max_pages": deepseek_max_pages_b or None,
            "dpi": deepseek_dpi_b,
            "custom_prompt": (deepseek_adv_prompt_b or None),
        }
    else: # Qwen3 VL
        st.subheader("Qwen3 VL Options (Batch)")
        col_b_q1, col_b_q2, col_b_q3 = st.columns(3)
        with col_b_q1:
            qwen_mode_b = st.radio("Extraction mode", ["markdown", "plain"], index=0, key="batch_qwen_mode")
        with col_b_q2:
            qwen_max_pages_b = st.number_input("Max pages", min_value=0, value=0, step=1, key="batch_qwen_max_pages")
        with col_b_q3:
            qwen_dpi_b = st.number_input("Render DPI", min_value=72, value=300, step=24, key="batch_qwen_dpi")

        col_b_q4, col_b_q5, col_b_q6 = st.columns(3)
        with col_b_q4:
            qwen_layout_b = st.selectbox("Layout", options=["auto", "single", "two_column"], index=0, key="batch_qwen_layout")
        with col_b_q5:
            qwen_lang_hint_b = st.text_input("Language hint (e.g., te)", value="", key="batch_qwen_lang_hint")
        with col_b_q6:
            qwen_enhance_b = st.checkbox("Enhance image", value=False, key="batch_qwen_enhance")

        col_b_q7, col_b_q8, col_b_q9 = st.columns(3)
        with col_b_q7:
            qwen_tiling_b = st.checkbox("High-res tiling (2x2)", value=False, key="batch_qwen_tiling")
        with col_b_q8:
            qwen_tile_rows_b = st.number_input("Tile rows", min_value=1, value=2, step=1, key="batch_qwen_tile_rows", disabled=not st.session_state.get("batch_qwen_tiling", False))
        with col_b_q9:
            qwen_tile_cols_b = st.number_input("Tile cols", min_value=1, value=2, step=1, key="batch_qwen_tile_cols", disabled=not st.session_state.get("batch_qwen_tiling", False))

        col_b_q10, col_b_q11 = st.columns(2)
        with col_b_q10:
            qwen_temperature_b = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1, format="%.1f", key="batch_qwen_temperature")
        with col_b_q11:
            qwen_single_req_b = st.checkbox("Combine tiles in one request", value=True, key="batch_qwen_multi_image_single_request")

        qwen_adv_prompt_b = st.text_area("Advanced prompt (optional)", value="", key="batch_qwen_adv_prompt")
        qwen_options_batch = {
            "mode": qwen_mode_b,
            "max_pages": qwen_max_pages_b or None,
            "dpi": qwen_dpi_b,
            "custom_prompt": (qwen_adv_prompt_b or None),
            "layout": qwen_layout_b,
            "language_hint": (qwen_lang_hint_b or None),
            "enhance_image": qwen_enhance_b,
            "highres_tiling": qwen_tiling_b,
            "tile_rows": qwen_tile_rows_b if qwen_tiling_b else 1,
            "tile_cols": qwen_tile_cols_b if qwen_tiling_b else 1,
            "temperature_override": qwen_temperature_b,
            "multi_image_single_request": qwen_single_req_b,
        }

    # Batch processing button
    st.subheader("Run Batch Multi‑Run Evaluation")
    run_batch = st.button("🔁 Run Multi‑Run for Uploaded PDFs", type="primary", disabled=not batch_pdfs)

    if run_batch and batch_pdfs:
        with st.spinner("Running multi‑run evaluations for all uploaded PDFs..."):
            try:
                runner = MultiRunRunner(num_runs=get_settings().multi_run_runs)
                reporter = MultiRunReporter()
            except Exception:
                runner = MultiRunRunner()
                reporter = MultiRunReporter()
            # Collect results across all PDFs for final consolidated report
            batch_results: List[Dict[str, Any]] = []

            for uploaded in batch_pdfs:
                pdf_name = uploaded.name
                pdf_basename = get_pdf_basename(pdf_name)
                gt_path = _find_ground_truth_for_pdf(pdf_name)

                with st.expander(f"📄 {pdf_name}", expanded=True):
                    if not gt_path or not os.path.exists(gt_path):
                        st.error("Ground truth not found by filename in ground_truth/. Skipping.")
                        continue

                    try:
                        with open(gt_path, 'r', encoding='utf-8') as gf:
                            gt_text_batch = gf.read()
                    except Exception as e:
                        st.error(f"Failed to read ground truth: {e}")
                        continue

                    file_bytes = uploaded.getvalue()

                    # Create a wrapper for the selected model's extraction function
                    if "Marker" in model_choice_batch:
                        extraction_func = create_extraction_wrapper(_submit_marker_job, _extract_markdown)
                        run_options = {
                            'model_name': 'Marker',
                            'file_bytes': file_bytes,
                            'filename': uploaded.name,
                            'mime_type': (uploaded.type or "application/pdf"),
                            'settings': settings,
                            **marker_options_batch
                        }
                    elif "Surya" in model_choice_batch:
                        extraction_func = create_extraction_wrapper(_submit_ocr_job, _extract_markdown)
                        run_options = {
                            'model_name': 'Surya',
                            'file_bytes': file_bytes,
                            'filename': uploaded.name,
                            'mime_type': (uploaded.type or "application/pdf"),
                            'settings': settings,
                            **ocr_options_batch
                        }
                    elif "Deepseek" in model_choice_batch:
                        extraction_func = create_extraction_wrapper(_submit_deepseek_job, _extract_markdown)
                        run_options = {
                            'model_name': 'Deepseek',
                            'file_bytes': file_bytes,
                            'filename': uploaded.name,
                            'mime_type': (uploaded.type or "application/pdf"),
                            'settings': get_deepinfra_settings(),
                            **deepseek_options_batch
                        }
                    else: # Qwen3 VL via OpenRouter
                        extraction_func = create_extraction_wrapper(_submit_qwen_job, _extract_markdown)
                        run_options = {
                            'model_name': 'Qwen3 VL 8B',
                            'file_bytes': file_bytes,
                            'filename': uploaded.name,
                            'mime_type': (uploaded.type or "application/pdf"),
                            'settings': get_openrouter_settings(),
                            **qwen_options_batch
                        }

                    # Add a delay between runs to avoid rate limiting
                    original_extraction = extraction_func
                    def extraction_with_delay(**params):
                        time.sleep(5)
                        return original_extraction(**params)
                    extraction_func = extraction_with_delay

                    status_area = st.empty()
                    def progress_cb(msg):
                        status_area.info(msg)

                    try:
                        summary = runner.execute_multi_run_evaluation(
                            pdf_basename=pdf_basename,
                            model_folder=model_folder_batch,
                            gt_text=gt_text_batch,
                            extraction_func=extraction_func,
                            extraction_params=run_options,
                            progress_callback=progress_cb
                        )
                    except Exception as e:
                        st.error(f"Multi‑run failed: {e}")
                        continue

                    # Load per‑run metrics to render table (no extra analysis)
                    run_metrics_list = []
                    for run_detail in summary.run_details:
                        if run_detail.get('status') != 'success':
                            continue
                        metrics_path = os.path.join(
                            runner.output_base_dir, model_folder_batch, pdf_basename,
                            run_detail['metrics_file']
                        )
                        if os.path.exists(metrics_path):
                            with open(metrics_path, 'r', encoding='utf-8') as f:
                                m = json.load(f)
                            from multi_run_evaluation import RunMetrics
                            run_metrics_list.append(RunMetrics(
                                run_id=int(m['run_id']),
                                wer=float(m['wer']),
                                mer=float(m['mer']),
                                wil=float(m['wil']),
                                cer=float(m['cer']),
                                lev_distance=int(m.get('lev_distance', 0)),
                                lev_norm=float(m.get('lev_norm', 0.0)),
                                structural_accuracy=m.get('structural_accuracy', {}),
                                structural_analysis=m.get('structural_analysis', {}),
                                completeness=float(m.get('completeness', 0.0)),
                                word_mismatches=m.get('word_mismatches', [])
                            ))

                    # Save standard comprehensive report for consistency on disk
                    try:
                        report_md_path, report_txt_path, _ = reporter.generate_comprehensive_report(
                            summary, run_metrics_list, model_folder_batch
                        )
                    except Exception:
                        report_md_path, report_txt_path = "", ""

                    # Display only the per‑run metrics table
                    st.markdown("**Per‑Run Metrics**")
                    try:
                        per_run_md = reporter.generate_per_run_metrics_table(run_metrics_list)
                        st.markdown(per_run_md)
                    except Exception as e:
                        st.warning(f"Could not render per‑run table: {e}")

                    # Quick downloads
                    dl1, dl2, dl3 = st.columns(3)
                    with dl1:
                        if report_md_path and os.path.exists(report_md_path):
                            with open(report_md_path, 'r', encoding='utf-8') as f:
                                md_content = f.read()
                            st.download_button("⬇️ Report (Markdown)", data=md_content, file_name=f"multirun_report_{pdf_basename}.md", mime="text/markdown")
                    with dl2:
                        # Summary JSON
                        from dataclasses import asdict
                        summary_dict = asdict(summary)
                        summary_dict['aggregated_metrics'] = {k: asdict(v) for k, v in summary.aggregated_metrics.items()}
                        st.download_button("⬇️ Summary (JSON)", data=json.dumps(summary_dict, indent=2, ensure_ascii=False), file_name=f"multirun_summary_{pdf_basename}.json", mime="application/json")
                    with dl3:
                        if report_txt_path and os.path.exists(report_txt_path):
                            with open(report_txt_path, 'r', encoding='utf-8') as f:
                                txt_content = f.read()
                                st.download_button("⬇️ Report (Text)", data=txt_content, file_name=f"multirun_report_{pdf_basename}.txt", mime="text/plain")

                    # (Removed) Per-file Final Report tab — consolidated report will be shown after batch completes

                    # Collect for overall final report after batch finishes
                    try:
                        batch_results.append({
                            'pdf_basename': pdf_basename,
                            'model_folder': model_folder_batch,
                            'summary': summary,
                            'run_metrics_list': run_metrics_list,
                        })
                    except Exception:
                        pass

            # After processing all PDFs, show a single consolidated Final Report tab
            if batch_results:
                try:
                    final_tab = st.tabs(["Final Report (Batch)"])[0]
                    with final_tab:
                        consolidated_lines = []
                        consolidated_lines.append("# 📦 Final Batch Report\n")
                        consolidated_lines.append(f"**Files Processed:** {len(batch_results)}  ")
                        consolidated_lines.append(f"**Model:** `{model_folder_batch}`  \n")

                        # Consolidated table of composite scores across PDFs
                        consolidated_lines.append("## Consolidated Scores (Per PDF)\n")
                        consolidated_lines.append("| PDF | Text Score | Structural Score | Overall Score | Overall CCI |")
                        consolidated_lines.append("|-----|------------|------------------|---------------|-------------|")
                        for item in batch_results:
                            pdf_name = item['pdf_basename']
                            summ = item['summary']
                            consolidated_lines.append(
                                f"| {pdf_name} | "
                                f"{((f'{summ.text_score:.4f}') if summ.text_score is not None else 'NA')} | "
                                f"{((f'{summ.structural_score:.4f}') if summ.structural_score is not None else 'NA')} | "
                                f"{((f'{summ.overall_score:.4f}') if summ.overall_score is not None else 'NA')} | "
                                f"{summ.overall_cci:.4f} |"
                            )

                        for item in batch_results:
                            pdf_name = item['pdf_basename']
                            summ = item['summary']
                            runs = item['run_metrics_list']
                            consolidated_lines.append("\n---\n")
                            consolidated_lines.append(f"## 📄 {pdf_name}\n")
                            consolidated_lines.append(f"**Total Runs:** {summ.total_runs}  ")
                            consolidated_lines.append(f"**Successful Runs:** {summ.successful_runs}  ")
                            consolidated_lines.append(f"**Overall CCI:** {summ.overall_cci:.4f}  ")
                            consolidated_lines.append(f"**Stability:** {summ.stability_interpretation}  \n")
                            # Per-file Aggregate scores quick table
                            consolidated_lines.append("### Aggregate Scores\n")
                            consolidated_lines.append("| Metric | Score |")
                            consolidated_lines.append("|--------|-------|")
                            consolidated_lines.append(f"| **Text Accuracy Score** | {((f'{summ.text_score:.4f}') if summ.text_score is not None else 'NA')} |")
                            consolidated_lines.append(f"| **Structural Score** | {((f'{summ.structural_score:.4f}') if summ.structural_score is not None else 'NA')} |")
                            consolidated_lines.append(f"| **Overall Extraction Score** | {((f'{summ.overall_score:.4f}') if summ.overall_score is not None else 'NA')} |\n")
                            consolidated_lines.append("### Per-Run Metrics\n")
                            consolidated_lines.append(reporter.generate_per_run_metrics_table(runs))
                            # (Omitted) Aggregated Metrics — per-run tables are sufficient

                        batch_md = "\n".join(consolidated_lines)
                        st.markdown(batch_md)
                        # Info expander with consolidated ideal ranges table
                        with st.expander("ℹ️ Ideal Evaluation Metric Ranges (Consolidated Table)"):
                            st.markdown(_ideal_ranges_table_md())

                        # Optional: download consolidated batch report
                        from datetime import datetime as _dt
                        ts = _dt.now().strftime('%Y%m%d_%H%M%S')
                        st.download_button(
                            label="⬇️ Download Final Batch Report (.md)",
                            data=batch_md,
                            file_name=f"final_batch_report_{ts}.md",
                            mime="text/markdown"
                        )
                except Exception as e:
                    st.warning(f"Could not render Final Batch Report: {e}")


def render_manual_section():
    """Render the Manual File Comparison section"""
    # Manual Comparison Section
    st.header("Manual File Comparison")
    st.caption("Upload ground truth and OCR output files separately for comparison")
    
    col_gt, col_ocr = st.columns(2)
    
    with col_gt:
        st.subheader("Ground Truth File")
        gt_file = st.file_uploader(
            "Upload Ground Truth (.md file)",
            type=["md"],
            key="gt_upload",
            help="Upload the ground truth markdown file"
        )
        if gt_file:
            st.success(f"✅ Ground truth loaded: {gt_file.name}")
            with st.expander("Preview Ground Truth"):
                gt_content = gt_file.getvalue().decode('utf-8')
                st.code(gt_content[:500] + "..." if len(gt_content) > 500 else gt_content, language="markdown")
    
    with col_ocr:
        st.subheader("OCR Output File")
        ocr_file = st.file_uploader(
            "Upload OCR Output (.md file)",
            type=["md"],
            key="ocr_upload",
            help="Upload the OCR output markdown file"
        )
        if ocr_file:
            st.success(f"✅ OCR output loaded: {ocr_file.name}")
            with st.expander("Preview OCR Output"):
                ocr_content = ocr_file.getvalue().decode('utf-8')
                st.code(ocr_content[:500] + "..." if len(ocr_content) > 500 else ocr_content, language="markdown")

    # End of Manual section
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Manual comparison controls
    if gt_file and ocr_file:
        st.subheader("Comparison Settings")
        
        col_settings, col_button = st.columns([2, 1])
        
        with col_settings:
            comparison_output_dir = st.text_input(
                "Output Directory",
                value="evaluation/results/manual",
                help="Directory where comparison results will be saved"
            )
        
        with col_button:
            st.write("")  # Add some spacing
            run_comparison = st.button("🔍 Run Comparison", type="primary")
        
        if run_comparison:
            with st.spinner("Running comparison..."):
                try:
                    import tempfile
                    from compare import evaluate_ocr_performance
                    
                    # Generate meaningful report name from file names
                    def generate_report_name(gt_filename, ocr_filename):
                        """Generate report name using first 5-8 characters of each file name"""
                        # Remove file extensions and get base names
                        gt_base = os.path.splitext(gt_filename)[0]
                        ocr_base = os.path.splitext(ocr_filename)[0]
                        gt_short = gt_base[:6] if len(gt_base) > 6 else gt_base
                        ocr_short = ocr_base[:6] if len(ocr_base) > 6 else ocr_base
                        return f"{gt_short}_{ocr_short}"
                    
                    custom_report_name = generate_report_name(gt_file.name, ocr_file.name)

                    # Create temporary files
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp_gt:
                        tmp_gt.write(gt_file.getvalue().decode('utf-8'))
                        tmp_gt_path = tmp_gt.name
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp_ocr:
                        tmp_ocr.write(ocr_file.getvalue().decode('utf-8'))
                        tmp_ocr_path = tmp_ocr.name
                    
                    try:
                        # Run comparison with custom report name using gt_text and ocr_text directly
                        os.makedirs(comparison_output_dir, exist_ok=True)
                        
                        # Use gt_text and ocr_text parameters to pass content directly
                        gt_text = gt_file.getvalue().decode('utf-8')
                        ocr_text = ocr_file.getvalue().decode('utf-8')
                        
                        # Create temporary OCR file (still needed for ocr_file parameter)
                        summary_txt_path, summary_md_path, chart_path = evaluate_ocr_performance(
                            gt_text=gt_text,
                            ocr_file=tmp_ocr_path,
                            output_dir=comparison_output_dir,
                            custom_name=custom_report_name,
                            display_model_name="Manual Comparison",
                            gt_display_name=gt_file.name,
                            ocr_display_name=ocr_file.name
                        )
                        
                        st.success("✅ Comparison completed successfully!")
                        
                        # Display results (organized)
                        tab_chart, tab_report = st.tabs(["Metrics Chart", "Full Report"])

                        with tab_chart:
                            if os.path.exists(chart_path):
                                st.image(chart_path, caption="Evaluation Metrics", use_container_width=True)
                            else:
                                st.warning("Chart not generated")

                        with tab_report:
                            st.subheader("Evaluation Summary")
                            if os.path.exists(summary_md_path):
                                try:
                                    with open(summary_md_path, 'r', encoding='utf-8') as f:
                                        md_content = f.read()
                                    st.markdown(md_content)
                                except Exception as e:
                                    st.warning(f"Could not read report: {e}")
                                
                                # Download buttons for both formats
                                col_dl1, col_dl2 = st.columns(2)
                                with col_dl1:
                                    st.download_button(
                                        label="📥 Download Markdown Report",
                                        data=md_content,
                                        file_name=f"comparison_report_{gt_file.name}_{ocr_file.name}.md",
                                        mime="text/markdown"
                                    )
                                with col_dl2:
                                    if os.path.exists(summary_txt_path):
                                        with open(summary_txt_path, 'r', encoding='utf-8') as f:
                                            txt_content = f.read()
                                        st.download_button(
                                            label="📥 Download Text Report",
                                            data=txt_content,
                                            file_name=f"comparison_report_{gt_file.name}_{ocr_file.name}.txt",
                                            mime="text/plain"
                                        )
                            elif os.path.exists(summary_txt_path):
                                with open(summary_txt_path, 'r', encoding='utf-8') as f:
                                    report_content = f.read()
                                st.text_area("Report", report_content, height=300)
                                
                                # Download button for text report
                                st.download_button(
                                    label="📥 Download Report",
                                    data=report_content,
                                    file_name=f"comparison_report_{gt_file.name}_{ocr_file.name}.txt",
                                    mime="text/plain"
                                )
                            else:
                                st.error("Report not generated")
                        
                        st.info(f"📊 Results saved to: {comparison_output_dir}")

                        # Visual Difference Display
                        st.markdown("---")
                        st.subheader("📋 Visual Difference Analysis")
                        
                        # Diff view options - default to side-by-side
                        diff_view = st.radio(
                            "Select diff view:",
                            ["Side-by-Side Comparison", "Unified Diff (GitHub-style)"],
                            horizontal=True,
                            index=0  # Default to side-by-side
                        )
                        
                        gt_content = gt_file.getvalue().decode('utf-8')
                        ocr_content = ocr_file.getvalue().decode('utf-8')
                        
                        if diff_view == "Side-by-Side Comparison":
                            st.markdown("**Side-by-Side Comparison**")
                            
                            # Split content into lines for comparison
                            gt_lines = gt_content.splitlines()
                            ocr_lines = ocr_content.splitlines()
                            
                            # Create side-by-side comparison with improved layout
                            col_gt_side, col_ocr_side = st.columns(2)
                            
                            with col_gt_side:
                                st.markdown(f"**Ground Truth ({gt_file.name})**")
                                
                                # Generate line-by-line diff for highlighting
                                matcher = difflib.SequenceMatcher(None, gt_lines, ocr_lines)
                                gt_html_lines = []
                                
                                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                                    if tag == 'equal':
                                        for i in range(i1, i2):
                                            line_escaped = gt_lines[i].replace('<', '<').replace('>', '>').replace('\n', '')
                                            gt_html_lines.append(f'<div style="padding: 4px 8px; line-height: 1.6; border-left: 3px solid #28a745;"><span style="color: #666; margin-right: 12px; font-weight: bold; min-width: 40px; display: inline-block;">{i+1:4d}</span><span>{line_escaped}</span></div>')
                                    elif tag == 'delete':
                                        for i in range(i1, i2):
                                            line_escaped = gt_lines[i].replace('<', '<').replace('>', '>').replace('\n', '')
                                            gt_html_lines.append(f'<div style="background:#ffeef0; padding: 4px 8px; line-height: 1.6; border-left: 3px solid #dc3545; text-decoration:line-through;"><span style="color: #666; margin-right: 12px; font-weight: bold; min-width: 40px; display: inline-block;">{i+1:4d}</span><span style="color: #721c24; text-decoration: line-through;">{line_escaped}</span></div>')
                                    elif tag == 'insert':
                                        # Add placeholder lines for insertions in OCR
                                        for j in range(j1, j2):
                                            gt_html_lines.append(f'<div style="background:#f8f9fa; padding: 4px 8px; line-height: 1.6; border-left: 3px solid #6c757d;"><span style="color: #666; margin-right: 12px; font-weight: bold; min-width: 40px; display: inline-block;">    </span><span style="color: #6c757d; font-style: italic;">+ Added in OCR</span></div>')
                                    elif tag == 'replace':
                                        for i in range(i1, i2):
                                            line_escaped = gt_lines[i].replace('<', '<').replace('>', '>').replace('\n', '')
                                            gt_html_lines.append(f'<div style="background:#fff3cd; padding: 4px 8px; line-height: 1.6; border-left: 3px solid #ffc107;"><span style="color: #666; margin-right: 12px; font-weight: bold; min-width: 40px; display: inline-block;">{i+1:4d}</span><span style="color: #856404;">{line_escaped}</span></div>')
                                
                                st.markdown(
                                    f'<div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; font-family: \'Courier New\', monospace; font-size: 13px; max-height: 600px; overflow-y: auto; margin: 8px 0;">{"".join(gt_html_lines)}</div>',
                                    unsafe_allow_html=True
                                )
                            
                            with col_ocr_side:
                                st.markdown(f"**OCR Output ({ocr_file.name})**")
                                
                                ocr_html_lines = []
                                
                                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                                    if tag == 'equal':
                                        for j in range(j1, j2):
                                            line_escaped = ocr_lines[j].replace('<', '<').replace('>', '>').replace('\n', '')
                                            ocr_html_lines.append(f'<div style="padding: 4px 8px; line-height: 1.6; border-left: 3px solid #28a745;"><span style="color: #666; margin-right: 12px; font-weight: bold; min-width: 40px; display: inline-block;">{j+1:4d}</span><span>{line_escaped}</span></div>')
                                    elif tag == 'insert':
                                        for j in range(j1, j2):
                                            line_escaped = ocr_lines[j].replace('<', '<').replace('>', '>').replace('\n', '')
                                            ocr_html_lines.append(f'<div style="background:#d1f4d0; padding: 4px 8px; line-height: 1.6; border-left: 3px solid #198754;"><span style="color: #666; margin-right: 12px; font-weight: bold; min-width: 40px; display: inline-block;">{j+1:4d}</span><span style="color: #0f5132; font-weight: bold;">+ {line_escaped}</span></div>')
                                    elif tag == 'delete':
                                        # Add placeholder lines for deletions in GT
                                        for i in range(i1, i2):
                                            ocr_html_lines.append(f'<div style="background:#f8f9fa; padding: 4px 8px; line-height: 1.6; border-left: 3px solid #6c757d;"><span style="color: #666; margin-right: 12px; font-weight: bold; min-width: 40px; display: inline-block;">    </span><span style="color: #6c757d; font-style: italic;">- Removed from GT</span></div>')
                                    elif tag == 'replace':
                                        for j in range(j1, j2):
                                            line_escaped = ocr_lines[j].replace('<', '<').replace('>', '>').replace('\n', '')
                                            ocr_html_lines.append(f'<div style="background:#fff3cd; padding: 4px 8px; line-height: 1.6; border-left: 3px solid #ffc107;"><span style="color: #666; margin-right: 12px; font-weight: bold; min-width: 40px; display: inline-block;">{j+1:4d}</span><span style="color: #856404;">{line_escaped}</span></div>')
                                
                                st.markdown(
                                    f'<div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; font-family: \'Courier New\', monospace; font-size: 13px; max-height: 600px; overflow-y: auto; margin: 8px 0;">{"".join(ocr_html_lines)}</div>',
                                    unsafe_allow_html=True
                                )
                            
                            # Diff statistics
                            st.markdown("**Difference Statistics:**")
                            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                            
                            total_ops = len(list(matcher.get_opcodes()))
                            equal_ops = sum(1 for tag, _, _, _, _ in matcher.get_opcodes() if tag == 'equal')
                            
                            with col_stats1:
                                st.metric("Total Lines (GT)", len(gt_lines))
                            with col_stats2:
                                st.metric("Total Lines (OCR)", len(ocr_lines))
                            with col_stats3:
                                similarity = matcher.ratio()
                                st.metric("Similarity", f"{similarity:.2%}")
                            with col_stats4:
                                st.metric("Matching Blocks", equal_ops)
                        
                        elif diff_view == "Unified Diff (GitHub-style)":
                            # Generate unified diff
                            diff_lines = list(difflib.unified_diff(
                                gt_content.splitlines(keepends=True),
                                ocr_content.splitlines(keepends=True),
                                fromfile=f'Ground Truth ({gt_file.name})',
                                tofile=f'OCR Output ({ocr_file.name})',
                                lineterm=''
                            ))
                            
                            if diff_lines:
                                # Format diff for better display
                                diff_html = []
                                for line in diff_lines:
                                    line_escaped = line.replace('<', '<').replace('>', '>')
                                    if line.startswith('---') or line.startswith('+++'):
                                        diff_html.append(f'<div style="color: #666; font-weight: bold; background-color: #f8f9fa; padding: 2px 8px; border-left: 4px solid #d1ecf1;">{line_escaped}</div>')
                                    elif line.startswith('@@'):
                                        diff_html.append(f'<div style="color: #0969da; background-color: #ddf4ff; padding: 2px 8px; border-left: 4px solid #0969da;">{line_escaped}</div>')
                                    elif line.startswith('+'):
                                        diff_html.append(f'<div style="background-color: #d1f4d0; color: #0f5132; padding: 2px 8px; border-left: 4px solid #198754;">{line_escaped}</div>')
                                    elif line.startswith('-'):
                                        diff_html.append(f'<div style="background-color: #f8d7da; color: #721c24; padding: 2px 8px; border-left: 4px solid #dc3545;">{line_escaped}</div>')
                                    else:
                                        diff_html.append(f'<div style="padding: 2px 8px;">{line_escaped}</div>')
                                
                                st.markdown(
                                    f'<div style="background-color: #f6f8fa; border: 1px solid #d1d9e0; border-radius: 6px; font-family: monospace; font-size: 12px; max-height: 500px; overflow-y: auto;">{"".join(diff_html)}</div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                st.info("✅ No differences found between the files!")
                        
                    finally:
                        # Clean up temporary files
                        if os.path.exists(tmp_gt_path):
                            os.remove(tmp_gt_path)
                        if os.path.exists(tmp_ocr_path):
                            os.remove(tmp_ocr_path)
                            
                except Exception as e:
                    st.error(f"❌ Comparison failed: {str(e)}")
    
    elif gt_file or ocr_file:
        st.info("ℹ️ Upload both ground truth and OCR output files to enable comparison")


# Helper: Ideal ranges table markdown for info expander
def _ideal_ranges_table_md() -> str:
    lines = []
    lines.append("**Interpretation rule:**")
    lines.append("- For error-based metrics, lower is better (→ 0)")
    lines.append("- For score-based metrics, higher is better (→ 1)\n")
    lines.append("| Metric | Type | Ideal Value | Very Good | Acceptable | Poor | What It Indicates |")
    lines.append("|--------|------|-------------|-----------|------------|------|-------------------|")
    lines.append("| WER (Word Error Rate) | Error | 0.00 | < 0.15 | 0.15–0.30 | > 0.30 | Word-level accuracy |")
    lines.append("| CER (Character Error Rate) | Error | 0.00 | < 0.05 | 0.05–0.20 | > 0.20 | Character-level accuracy |")
    lines.append("| MER (Match Error Rate) | Error | 0.00 | < 0.20 | 0.20–0.35 | > 0.35 | Overall edit errors |")
    lines.append("| WIL (Word Information Lost) | Error | 0.00 | < 0.20 | 0.20–0.40 | > 0.40 | Information loss |")
    lines.append("| LEV-DIST (Levenshtein chars) | Count | 0 | < 0.10·|GT| | 0.10–0.25·|GT| | > 0.25·|GT| | Raw character edit distance |")
    lines.append("| Completeness | Score | 1.00 | > 0.95 | 0.85–0.95 | < 0.85 | Content coverage |")
    lines.append("| Heading Alignment | Score | 1.00 | > 0.90 | 0.75–0.90 | < 0.75 | Heading correctness |")
    lines.append("| List Accuracy | Score | 1.00 | > 0.90 | 0.75–0.90 | < 0.75 | List structure |")
    lines.append("| Table Preservation | Score | 1.00 | > 0.85 | 0.65–0.85 | < 0.65 | Table integrity |")
    lines.append("| Link Correctness | Score | 1.00 | > 0.95 | 0.85–0.95 | < 0.85 | URL extraction |")
    lines.append("| Section Ordering | Score | 1.00 | > 0.90 | 0.75–0.90 | < 0.75 | Logical flow |")
    lines.append("| Text Score | Score | 1.00 | > 0.90 | 0.80–0.90 | < 0.80 | Overall text accuracy |")
    lines.append("| Structural Score | Score | 1.00 | > 0.90 | 0.75–0.90 | < 0.75 | Structural fidelity |")
    lines.append("| Overall Score | Score | 1.00 | > 0.90 | 0.80–0.90 | < 0.80 | Production readiness |")
    lines.append("| Consistency Confidence Index (CCI) | Score | 1.00 | > 0.90 | 0.80–0.90 | < 0.80 | Run stability |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

