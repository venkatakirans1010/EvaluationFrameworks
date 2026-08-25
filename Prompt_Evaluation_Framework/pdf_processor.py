"""
PDF Processing Module
Handles PDF file upload and text extraction, including image extraction for vision-capable models
"""
import PyPDF2
import pdfplumber
import base64
from io import BytesIO
from typing import Optional, Dict, Any, List, Tuple
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def extract_text_from_pdf(pdf_file) -> Optional[str]:
    """
    Extract text from a PDF file using multiple methods for better coverage
    
    Args:
        pdf_file: File-like object or bytes containing PDF data
    
    Returns:
        Extracted text as string, or None if extraction fails
    """
    text_content = []
    
    # Method 1: Try pdfplumber (better for complex layouts)
    try:
        if hasattr(pdf_file, 'read'):
            pdf_file.seek(0)  # Reset file pointer
            pdf_bytes = pdf_file.read()
        else:
            pdf_bytes = pdf_file
        
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
    except Exception as e:
        print(f"pdfplumber extraction failed: {e}")
    
    # Method 2: Fallback to PyPDF2 if pdfplumber didn't work or extracted little text
    if not text_content or len(' '.join(text_content)) < 100:
        try:
            if hasattr(pdf_file, 'read'):
                pdf_file.seek(0)
                pdf_bytes = pdf_file.read()
            else:
                pdf_bytes = pdf_file
            
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
    
    if text_content:
        return '\n\n'.join(text_content)
    return None

def extract_images_from_pdf(pdf_file) -> List[Tuple[bytes, str]]:
    """
    Extract images from PDF pages by rendering pages as images
    This is better for vision models as it captures the entire page layout
    
    Args:
        pdf_file: File-like object or bytes containing PDF data
    
    Returns:
        List of tuples (image_bytes, mime_type) for each page rendered as image
    """
    images = []
    
    try:
        if hasattr(pdf_file, 'read'):
            pdf_file.seek(0)
            pdf_bytes = pdf_file.read()
        else:
            pdf_bytes = pdf_file
        
        # Method 1: Render pages as images using pdfplumber (best for vision models)
        # Try pdfplumber's to_image() first (requires pdfplumber with image support)
        if PIL_AVAILABLE:
            try:
                with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                    for page_num, page in enumerate(pdf.pages[:10]):  # Limit to 10 pages
                        try:
                            # Try to render page as image using pdfplumber's to_image()
                            # This requires pdfplumber to be installed with image rendering support
                            if hasattr(page, 'to_image'):
                                try:
                                    page_image = page.to_image(resolution=200)  # 200 DPI for reasonable size
                                    # Convert to PIL Image
                                    pil_image = page_image.original
                                    # Convert to bytes
                                    img_buffer = BytesIO()
                                    pil_image.save(img_buffer, format='PNG')
                                    img_bytes = img_buffer.getvalue()
                                    images.append((img_bytes, "image/png"))
                                except (AttributeError, ImportError, Exception) as e:
                                    # to_image() might not be available or might fail
                                    # This is expected if pdfplumber doesn't have image rendering support
                                    # Fall through to other methods
                                    if page_num == 0:  # Only log once
                                        print(f"pdfplumber to_image() not available or failed: {e}")
                                    break  # If first page fails, others will too
                        except Exception as e:
                            print(f"Error rendering page {page_num}: {e}")
            except Exception as e:
                print(f"pdfplumber page rendering failed: {e}")
        
        # Method 1b: Try pdf2image as alternative (requires poppler)
        if not images:
            try:
                from pdf2image import convert_from_bytes
                pil_images = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=min(10, len(pdf_bytes) // 10000))
                for pil_img in pil_images:
                    img_buffer = BytesIO()
                    pil_img.save(img_buffer, format='PNG')
                    img_bytes = img_buffer.getvalue()
                    images.append((img_bytes, "image/png"))
            except ImportError:
                # pdf2image not installed, skip
                pass
            except Exception as e:
                print(f"pdf2image conversion failed: {e}")
        
        # Method 2: Extract embedded images using PyPDF2
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    # Get resources and handle IndirectObject properly
                    resources = page.get('/Resources')
                    if resources:
                        resources_obj = resources.get_object() if hasattr(resources, 'get_object') else resources
                        
                        # Check for XObject in resources
                        if isinstance(resources_obj, dict) and '/XObject' in resources_obj:
                            xObject_ref = resources_obj['/XObject']
                            xObject = xObject_ref.get_object() if hasattr(xObject_ref, 'get_object') else xObject_ref
                            
                            if isinstance(xObject, dict):
                                for obj_name in xObject:
                                    try:
                                        obj_ref = xObject[obj_name]
                                        obj = obj_ref.get_object() if hasattr(obj_ref, 'get_object') else obj_ref
                                        
                                        # Check if it's an image
                                        if isinstance(obj, dict) and obj.get('/Subtype') == '/Image':
                                            try:
                                                size = (obj.get('/Width', 0), obj.get('/Height', 0))
                                                if size[0] > 0 and size[1] > 0:
                                                    data = obj.get_data()
                                                    
                                                    # Determine color space
                                                    color_space = obj.get('/ColorSpace')
                                                    if isinstance(color_space, list):
                                                        color_space = color_space[0]
                                                    
                                                    # Get color space name if it's an IndirectObject
                                                    if hasattr(color_space, 'get_object'):
                                                        color_space = color_space.get_object()
                                                    if isinstance(color_space, list) and len(color_space) > 0:
                                                        color_space = color_space[0]
                                                    
                                                    if color_space == '/DeviceRGB':
                                                        mode = "RGB"
                                                    elif color_space == '/DeviceGray':
                                                        mode = "L"
                                                    else:
                                                        mode = "RGB"  # Default to RGB
                                                    
                                                    # Convert to PNG format for better compatibility
                                                    if PIL_AVAILABLE:
                                                        try:
                                                            img = Image.frombytes(mode, size, data)
                                                            img_buffer = BytesIO()
                                                            img.save(img_buffer, format='PNG')
                                                            img_bytes = img_buffer.getvalue()
                                                            images.append((img_bytes, "image/png"))
                                                        except Exception as e:
                                                            print(f"Error converting image to PNG: {e}")
                                                    else:
                                                        # Fallback: use raw data
                                                        images.append((data, "image/jpeg"))
                                            except Exception as e:
                                                print(f"Error extracting image object {obj_name}: {e}")
                                    except Exception as e:
                                        print(f"Error processing object {obj_name}: {e}")
                except Exception as e:
                    print(f"Error processing page {page_num}: {e}")
        except Exception as e:
            print(f"PyPDF2 image extraction failed: {e}")
        
        # Method 3: If no images found yet, try rendering pages using PyPDF2 + PIL
        # This is a fallback for scanned PDFs where the whole page needs to be rendered
        if not images and PIL_AVAILABLE:
            try:
                pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        # Try to render the page
                        # Note: PyPDF2 doesn't have built-in rendering, but we can try
                        # For now, if we get here and have no images, we'll handle it in the calling function
                        pass
                    except Exception as e:
                        print(f"Error processing page {page_num} for rendering: {e}")
            except Exception as e:
                print(f"PyPDF2 page rendering attempt failed: {e}")
        
    except Exception as e:
        print(f"Image extraction failed: {e}")
    
    return images

def is_vision_capable_model(model: str) -> bool:
    """
    Check if a model supports vision/image inputs
    
    Args:
        model: Model name
    
    Returns:
        True if model supports vision, False otherwise
    """
    vision_models = [
        'gpt-4-vision', 'gpt-4o', 'gpt-4-turbo', 'gpt-4-turbo-preview', 'gpt-4o-mini',
        'gpt-5', 'gpt5',
        'claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku', 'claude-3-5-sonnet',
        'claude-3-5-haiku', 'claude-3-7-sonnet', 'claude-3.5-sonnet', 'claude-3.7-sonnet',
        'claude-haiku-4-5-20251001', 'claude-4',
        'gemini-pro-vision', 'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-2.0-flash-exp',
        'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-pro', 'gemini-flash',
        'llama-3.1-405b-vision', 'llama-3.1-70b-vision'
    ]
    
    # Also check for common patterns
    vision_patterns = [
        'gemini',  # Most Gemini models support vision
        'gpt-4',   # GPT-4 models generally support vision
        'claude-3', # Claude 3 models support vision
        'vision'   # Any model with "vision" in the name
    ]
    
    model_lower = model.lower()
    
    # Check exact matches first
    if any(vision_model in model_lower for vision_model in vision_models):
        return True
    
    # Check patterns (but be more careful - not all GPT-4 or Gemini models support vision)
    # For Gemini, most recent models support vision
    if 'gemini' in model_lower and ('2.5' in model_lower or '2.0' in model_lower or '1.5' in model_lower or 'pro-vision' in model_lower):
        return True
    
    # For GPT-4, check for vision-capable variants
    if 'gpt-4' in model_lower and ('vision' in model_lower or 'turbo' in model_lower or 'gpt-4o' in model_lower):
        return True
    
    # GPT-5 models support vision
    if 'gpt-5' in model_lower or 'gpt5' in model_lower:
        return True
    
    # Claude 3 and Claude 4 models generally support vision
    if 'claude-3' in model_lower or 'claude-4' in model_lower:
        return True
    
    return False

def extract_text_from_pdf_with_llm(
    pdf_file,
    llm_router,
    model_config: Dict[str, Any],
    api_keys: Dict[str, str],
    extraction_prompt: str = None
) -> Optional[str]:
    """
    Extract text from a PDF using an LLM, handling both text and images
    
    Args:
        pdf_file: File-like object or bytes containing PDF data
        llm_router: LLMRouter instance
        model_config: Model configuration dictionary
        api_keys: Dictionary of API keys
        extraction_prompt: Custom prompt for extraction (optional)
    
    Returns:
        Extracted text as string, or None if extraction fails
    """
    try:
        # Read PDF bytes
        if hasattr(pdf_file, 'read'):
            pdf_file.seek(0)
            pdf_bytes = pdf_file.read()
        else:
            pdf_bytes = pdf_file
        
        # Default extraction prompt if not provided
        if not extraction_prompt:
            extraction_prompt = """Extract all text content from this PDF document. 
Return only the extracted text content in a clean, readable format. 
Preserve the structure, formatting, and organization of the original document.
Do not add any explanations or additional text - just return the extracted content."""
        
        # Get model info
        provider = model_config.get('provider', 'routellm')
        model = model_config.get('model', '')
        api_key = api_keys.get(provider, '')
        
        if not api_key:
            return None
        
        # Check if model supports vision
        supports_vision = is_vision_capable_model(model)
        
        # Build content for LLM - USE ONLY LLM, NO TRADITIONAL EXTRACTION
        if supports_vision:
            # Use vision-capable model - send PDF/images directly to LLM
            # Extract images from PDF for vision models
            if hasattr(pdf_file, 'seek'):
                pdf_file.seek(0)
            pdf_images = extract_images_from_pdf(pdf_file)
            
            if pdf_images:
                # Extract embedded images or rendered page images and send them to vision model
                image_base64_list = []
                for img_bytes, mime_type in pdf_images[:10]:  # Limit to 10 images to avoid token limits
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    image_base64_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{img_base64}"
                        }
                    })
                
                # Build prompt - no mention of traditional extraction
                text_prompt = extraction_prompt
                text_prompt += f"\n\nExtract ALL visible text from the {len(image_base64_list)} image(s) provided. Return only the extracted text content in a clean, readable format. Preserve the structure and organization of the document."
                
                # Create content array with text and images
                content = [{"type": "text", "text": text_prompt}]
                content.extend(image_base64_list)
            else:
                # No embedded images found - need to render PDF pages as images
                # This is necessary for both scanned PDFs and normal PDFs with text
                page_images = []
                
                # Try pdf2image first (most reliable)
                try:
                    from pdf2image import convert_from_bytes
                    pil_images = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=min(10, 100))
                    for pil_img in pil_images:
                        img_buffer = BytesIO()
                        pil_img.save(img_buffer, format='PNG')
                        img_bytes = img_buffer.getvalue()
                        page_images.append((img_bytes, "image/png"))
                except ImportError:
                    # pdf2image not available, try pdfplumber
                    pass
                except Exception as e:
                    print(f"pdf2image conversion failed: {e}")
                
                # Fallback to pdfplumber's to_image() if pdf2image failed
                if not page_images and PIL_AVAILABLE:
                    try:
                        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                            for page_num, page in enumerate(pdf.pages[:10]):  # Limit to 10 pages
                                try:
                                    if hasattr(page, 'to_image'):
                                        page_image = page.to_image(resolution=200)
                                        pil_image = page_image.original
                                        img_buffer = BytesIO()
                                        pil_image.save(img_buffer, format='PNG')
                                        img_bytes = img_buffer.getvalue()
                                        page_images.append((img_bytes, "image/png"))
                                except Exception as e:
                                    if page_num == 0:  # Only log once
                                        print(f"pdfplumber to_image() failed: {e}")
                                    break  # If first page fails, others will too
                    except Exception as e:
                        print(f"Error rendering PDF pages with pdfplumber: {e}")
                
                if page_images:
                    # Successfully rendered pages as images
                    image_base64_list = []
                    for img_bytes, mime_type in page_images:
                        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                        image_base64_list.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{img_base64}"
                            }
                        })
                    
                    text_prompt = extraction_prompt
                    text_prompt += f"\n\nExtract ALL visible text from the {len(image_base64_list)} page image(s) provided. Return only the extracted text content in a clean, readable format. Preserve the structure and organization of the document."
                    
                    content = [{"type": "text", "text": text_prompt}]
                    content.extend(image_base64_list)
                else:
                    # Could not render pages - this is a problem
                    # Raise an error to inform the user
                    raise Exception(
                        "Unable to render PDF pages as images. "
                        "Please ensure pdf2image is installed with poppler, or pdfplumber has image rendering support. "
                        "For PDFs with text, you may want to use traditional extraction instead."
                    )
            
            # Call LLM with vision support
            result = llm_router.call_with_vision(
                content=content,
                model=model,
                provider=provider,
                api_key=api_key,
                temperature=model_config.get('temperature', 0.3),
                top_p=model_config.get('top_p', 1.0),
                max_tokens=model_config.get('max_tokens', 4000)
            )
        else:
            # Text-only model - still use LLM only, but we can't send images
            # For non-vision models, we'll inform the LLM about the PDF
            # Extract images if available (for information only)
            if hasattr(pdf_file, 'seek'):
                pdf_file.seek(0)
            pdf_images = extract_images_from_pdf(pdf_file)
            
            # Build prompt for text-only model
            # Note: We cannot send images to text-only models, so we inform them
            if pdf_images:
                prompt = f"""{extraction_prompt}

This PDF document contains {len(pdf_images)} image(s). 

Note: Your selected model ({model}) does not support vision/image processing. 
Please indicate that image-based PDF processing is not supported by this model.
For image-based PDFs, please use a vision-capable model (e.g., GPT-4 Vision, Claude 3, Gemini Pro Vision)."""
            else:
                prompt = f"""{extraction_prompt}

Extract ALL visible text from this PDF document. Return only the extracted text content in a clean, readable format.

Note: If this is an image-based PDF, your model may not be able to process it as it does not support vision processing."""
            
            # Extract parameters
            temperature = model_config.get('temperature', 0.3)
            top_p = model_config.get('top_p', 1.0)
            max_tokens = model_config.get('max_tokens', 4000)
            
            # Route to appropriate provider (text-only API)
            if provider == "routellm":
                result = llm_router.call_routellm(
                    prompt, model, api_key, temperature, top_p, max_tokens
                )
            elif provider == "openai":
                result = llm_router.call_openai(
                    prompt, model, api_key, temperature, top_p, max_tokens
                )
            elif provider == "anthropic":
                result = llm_router.call_anthropic(
                    prompt, model, api_key, temperature, top_p, max_tokens
                )
            else:
                return None
        
        if result.get('success') and result.get('response'):
            response_text = result.get('response', '').strip()
            if response_text:
                # Try to parse JSON response (some models return structured JSON)
                try:
                    import json
                    # Check if response is JSON
                    if response_text.strip().startswith('{') or response_text.strip().startswith('['):
                        # Try to extract JSON from markdown code blocks if present
                        json_text = response_text
                        if '```json' in response_text:
                            # Extract JSON from markdown code block
                            start = response_text.find('```json') + 7
                            end = response_text.find('```', start)
                            if end > start:
                                json_text = response_text[start:end].strip()
                        elif '```' in response_text:
                            # Extract from generic code block
                            start = response_text.find('```') + 3
                            end = response_text.find('```', start)
                            if end > start:
                                json_text = response_text[start:end].strip()
                        
                        # Parse JSON
                        parsed_json = json.loads(json_text)
                        
                        # Extract text content from common JSON structures
                        if isinstance(parsed_json, dict):
                            # Try common fields for extracted text
                            extracted_text = (
                                parsed_json.get('description') or
                                parsed_json.get('content') or
                                parsed_json.get('text') or
                                parsed_json.get('extracted_text') or
                                parsed_json.get('body')
                            )
                            
                            if extracted_text:
                                # Clean up the extracted text
                                # Remove references to "no text extracted" messages
                                extracted_text = extracted_text.replace(
                                    "No text could be extracted using traditional methods",
                                    ""
                                ).replace(
                                    "No text could be extracted using traditional text extraction methods",
                                    ""
                                ).replace(
                                    "Image-based PDF:",
                                    ""
                                ).replace(
                                    "This document appears to be a scanned image or an image-only PDF.",
                                    ""
                                ).strip()
                                
                                # Remove bracketed prefixes like "[Image-based PDF: ...]"
                                if extracted_text.startswith('[') and ']' in extracted_text:
                                    # Find the closing bracket and extract content after it
                                    bracket_end = extracted_text.find(']')
                                    if bracket_end > 0:
                                        extracted_text = extracted_text[bracket_end + 1:].strip()
                                
                                # Clean up any remaining bracketed content at the start
                                while extracted_text.startswith('[') and ']' in extracted_text[:200]:
                                    bracket_end = extracted_text.find(']')
                                    if bracket_end > 0:
                                        extracted_text = extracted_text[bracket_end + 1:].strip()
                                    else:
                                        break
                                
                                # If we have a title, prepend it (but avoid duplicates)
                                title = parsed_json.get('title', '')
                                if title and title not in extracted_text and title != "Image-based PDF Content":
                                    extracted_text = f"{title}\n\n{extracted_text}"
                                
                                # Return extracted text (no traditional extraction to combine)
                                
                                return extracted_text
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    # Not JSON or parsing failed, continue with plain text processing
                    pass
                
                # For plain text responses, clean up any error messages
                cleaned_text = response_text
                # Remove common error message patterns
                error_patterns = [
                    "No text could be extracted using traditional methods",
                    "No text could be extracted using traditional text extraction methods",
                    "This document appears to be a scanned image",
                    "image-based PDF"
                ]
                
                for pattern in error_patterns:
                    if pattern.lower() in cleaned_text.lower():
                        # Try to extract the actual content after the error message
                        # Look for content after common separators
                        separators = [":", "-", "\n\n", ". "]
                        for sep in separators:
                            if pattern.lower() in cleaned_text.lower():
                                parts = cleaned_text.split(sep, 1)
                                if len(parts) > 1 and len(parts[1].strip()) > 50:
                                    cleaned_text = parts[1].strip()
                                    break
                
                # Return cleaned text (no traditional extraction to combine)
                return cleaned_text
            else:
                # Empty response - log the error if available
                error_msg = result.get('error', 'Empty response from LLM')
                print(f"LLM extraction failed: {error_msg}")
                raise Exception(f"LLM returned empty response: {error_msg}")
        else:
            # API call failed
            error_msg = result.get('error', 'Unknown error from LLM API')
            print(f"LLM extraction failed: {error_msg}")
            raise Exception(f"LLM API error: {error_msg}")
        
    except Exception as e:
        # Re-raise with more context
        error_message = str(e)
        if "LLM" not in error_message and "API" not in error_message:
            raise Exception(f"PDF extraction error: {error_message}")
        raise
