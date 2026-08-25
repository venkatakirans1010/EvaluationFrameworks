"""
AI test case generator.
Uses Gemini 2.5 Flash to generate UI test cases from Jira issue details and RAG context.
"""

import logging
import re
import time
from typing import Dict, Any

import google.generativeai as genai

from config.settings import get_gemini_api_key

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=get_gemini_api_key())

# Prompt template - easy to tweak for experimentation
PROMPT_TEMPLATE = """You are a senior QA engineer with extensive experience in UI testing and test case design.

**Jira Story Details:**

**Story ID:** {story_id}
**Summary:** {summary}

**Description:**
{description}

**Acceptance Criteria:**
{acceptance}

**Top Comments:**
{comments}

**Attachments:**
{attachments}

**Supporting Documents:**
{context_text}

**Instructions:**

Generate comprehensive UI test cases based on the Jira story details and supporting documents above. 

**Output Requirements:**
- Produce UI test cases ONLY (no API, integration, or unit tests)
- Format as a Markdown table with the following columns:
  - Test ID (e.g., TC-001, TC-002, etc.)
  - Test Title (clear, concise description)
  - Preconditions (what must be true before test execution)
  - Steps (atomic, numbered steps like "1. Navigate to...", "2. Click on...")
  - Expected Result (clear, measurable outcome)
  - Type (one of: Positive, Negative, Boundary)
  - Priority (one of: High, Medium, Low)
  - Linked Requirements (reference to acceptance criteria, requirements from supporting documents, or document section/chunk numbers when applicable)

**Guidelines:**
- Generate up to {max_cases} test cases, but create only as many as necessary to thoroughly cover the functionality
- Each test step must be atomic and numbered (1., 2., 3., etc.)
- Ensure test cases cover positive scenarios, negative scenarios, and boundary conditions
- Link test cases to relevant acceptance criteria, requirements from supporting documents, or specific document sections/chunks when the test case is derived from uploaded documents
- When a test case is based on information from the Supporting Documents section, explicitly reference the document source or chunk in the Linked Requirements column (e.g., "From Supporting Document: [chunk/section reference]" or "Requirement from uploaded document")
- Keep test titles clear and descriptive
- Make expected results specific and measurable

**Output Format:**
Start with a brief introduction (1-2 sentences) explaining the test coverage approach, then provide the Markdown table."""


def generate_ui_test_cases(jira_story: Dict[str, Any], context_text: str, max_cases: int = 50) -> str:
    """
    Generate UI test cases using Gemini 2.5 Flash based on Jira story and RAG context.
    
    Args:
        jira_story: Dictionary containing Jira story details with keys:
                    id, summary, description, acceptance, comments, attachments
        context_text: Retrieved context from RAG engine (supporting documents)
        max_cases: Maximum number of test cases to generate (default: 50)
        
    Returns:
        str: Raw Markdown string containing test cases in table format
    """
    # Extract story details
    story_id = jira_story.get('id', 'Unknown')
    summary = jira_story.get('summary', '')
    description = jira_story.get('description', 'No description provided.')
    acceptance = jira_story.get('acceptance', 'No acceptance criteria provided.')
    
    # Format comments (limit to top 5 most relevant)
    comments_list = jira_story.get('comments', [])
    if comments_list:
        comments = '\n'.join([f"- {comment}" for comment in comments_list[:5]])
    else:
        comments = "No comments available."
    
    # Format attachments (just names)
    attachments_list = jira_story.get('attachments', [])
    if attachments_list:
        attachment_names = [att.get('filename', 'Unknown') for att in attachments_list]
        attachments = '\n'.join([f"- {name}" for name in attachment_names])
    else:
        attachments = "No attachments available."
    
    # Handle empty context
    if not context_text or not context_text.strip():
        context_text = "No supporting documents available."
    
    # Truncate context if too long to avoid token limits and safety issues
    # Keep last 8000 characters of context (roughly 2000 tokens)
    max_context_length = 8000
    if len(context_text) > max_context_length:
        logger.warning(f"Context text too long ({len(context_text)} chars), truncating to {max_context_length} chars")
        context_text = "... [earlier content truncated] ...\n\n" + context_text[-max_context_length:]
    
    # Build prompt
    prompt = PROMPT_TEMPLATE.format(
        story_id=story_id,
        summary=summary,
        description=description,
        acceptance=acceptance,
        comments=comments,
        attachments=attachments,
        context_text=context_text,
        max_cases=max_cases
    )
    
    # Log prompt length for debugging
    prompt_length = len(prompt)
    logger.info(f"Generated prompt with {prompt_length} characters for story {story_id}")
    logger.debug(f"Prompt preview (first 200 chars): {prompt[:200]}...")
    
    try:
        # Initialize Gemini model with relaxed safety settings for technical content
        # Use BLOCK_ONLY_HIGH to be less restrictive for technical documentation
        safety_settings = [
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH
            },
            {
                "category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH
            },
        ]
        
        # Prioritize free-tier models - these have better quota limits
        # Free tier models: gemini-1.5-flash, gemini-pro
        # Experimental models (like gemini-2.5-pro-exp) may not be available on free tier
        
        # First, list available models to see what's actually available
        available_models = []
        free_tier_models = []
        other_models = []
        
        try:
            logger.info("Listing available Gemini models...")
            for model_info in genai.list_models():
                if 'generateContent' in model_info.supported_generation_methods:
                    model_name = model_info.name
                    # Extract short name for filtering
                    short_name = model_name.split('/')[-1] if '/' in model_name else model_name
                    
                    # Filter out experimental models that aren't available on free tier
                    if 'exp' in short_name.lower() or '2.5' in short_name.lower():
                        logger.debug(f"Skipping experimental model: {model_name} (may not be on free tier)")
                        continue
                    
                    available_models.append(model_name)
                    
                    # Prioritize free-tier friendly models
                    if 'flash' in short_name.lower() or 'pro' in short_name.lower():
                        if 'flash' in short_name.lower():
                            free_tier_models.insert(0, model_name)  # Flash models first (cheaper)
                        else:
                            free_tier_models.append(model_name)
                    else:
                        other_models.append(model_name)
                    
                    logger.info(f"Found available model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not list models: {str(e)}. Will try free-tier model names.")
        
        # Build model list: prioritize free-tier models, exclude experimental
        model_names_to_try = []
        
        # First, add free-tier models (flash first, then pro)
        if free_tier_models:
            model_names_to_try.extend(free_tier_models)
        
        # Then add other non-experimental models
        if other_models:
            model_names_to_try.extend(other_models)
        
        # If we have discovered models, also try without "models/" prefix
        for model_name in available_models:
            if model_name.startswith('models/'):
                short_name = model_name.replace('models/', '')
                if short_name not in model_names_to_try and 'exp' not in short_name.lower():
                    model_names_to_try.append(short_name)
        
        # Fallback to known free-tier models if list_models didn't work
        if not model_names_to_try:
            logger.info("Using fallback free-tier model list")
            model_names_to_try = [
                'models/gemini-1.5-flash',    # Free tier - Flash (preferred)
                'gemini-1.5-flash',            # Without prefix
                'models/gemini-pro',           # Free tier - Pro (legacy)
                'gemini-pro',                  # Without prefix
            ]
        
        logger.info(f"Will try models in this order: {', '.join(model_names_to_try[:5])}...")
        
        model = None
        last_error = None
        
        for model_name in model_names_to_try:
            try:
                logger.info(f"Trying model: {model_name}")
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config={
                        'temperature': 0.2,  # Conservative temperature for determinism
                        'max_output_tokens': 8192,  # Appropriate token limit for test cases
                        'top_p': 0.95,
                    },
                    safety_settings=safety_settings
                )
                # If we get here, model initialization succeeded
                logger.info(f"Successfully initialized model: {model_name}")
                break
            except Exception as e:
                last_error = e
                error_msg = str(e)
                # Don't log 404 errors as warnings for all models, just skip
                if '404' in error_msg or 'not found' in error_msg.lower():
                    logger.debug(f"Model {model_name} not found: {error_msg}")
                else:
                    logger.warning(f"Model {model_name} error: {error_msg}")
                continue
        
        if model is None:
            error_details = f"Could not initialize any Gemini model."
            if last_error:
                error_details += f" Last error: {str(last_error)}"
            if available_models:
                error_details += f" Available models were: {', '.join(available_models)}"
            raise ValueError(error_details)
        
        # Generate test cases with retry logic for quota errors
        logger.info(f"Calling Gemini model for story {story_id}...")
        
        # Use full prompt with RAG context if available, otherwise use simplified prompt
        # Check if context_text contains actual content (not just placeholder)
        has_rag_context = context_text and context_text.strip() and context_text.strip() != "No supporting documents available."
        
        if has_rag_context:
            logger.info(f"✅ Using full prompt with RAG context ({len(context_text)} chars)")
            logger.info(f"📄 RAG context preview (first 300 chars): {context_text[:300]}...")
            prompt_to_use = prompt
        else:
            logger.info("No RAG context available, using simplified prompt to save tokens")
            # Use simplified prompt only when there's no RAG context
            prompt_to_use = PROMPT_TEMPLATE.format(
                story_id=story_id,
                summary=summary[:500] if len(summary) > 500 else summary,  # Truncate summary
                description=description[:1500] if len(description) > 1500 else description,  # Truncate description
                acceptance=acceptance[:800] if len(acceptance) > 800 else acceptance,  # Truncate acceptance
                comments=comments[:400] if len(comments) > 400 else comments,  # Truncate comments
                attachments=attachments,
                context_text="No supporting documents available.",
                max_cases=min(max_cases, 20)  # Reduce max cases for free tier
            )
        
        response = None
        max_retries = 3
        retry_delay = 5  # Start with 5 seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} to generate content...")
                response = model.generate_content(prompt_to_use)
                break  # Success, exit retry loop
            except Exception as error:
                error_msg = str(error)
                error_lower = error_msg.lower()
                
                # Check for quota errors
                if '429' in error_msg or 'quota' in error_lower or 'rate limit' in error_lower:
                    if attempt < max_retries - 1:
                        # Extract retry delay from error if available
                        if 'retry' in error_lower and 'seconds' in error_lower:
                            try:
                                delay_match = re.search(r'(\d+\.?\d*)\s*seconds?', error_msg, re.IGNORECASE)
                                if delay_match:
                                    retry_delay = float(delay_match.group(1)) + 2  # Add buffer
                            except:
                                pass
                        
                        logger.warning(
                            f"Quota exceeded (attempt {attempt + 1}/{max_retries}). "
                            f"Waiting {retry_delay} seconds before retry..."
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        # Last attempt failed
                        raise ValueError(
                            f"Quota exceeded after {max_retries} attempts. "
                            f"This API key has reached its free tier limits. "
                            f"Please check: https://ai.dev/usage?tab=rate-limit "
                            f"Error: {error_msg[:500]}"
                        )
                else:
                    # Non-quota error, don't retry
                    logger.error(f"Generation failed: {error_msg}")
                    raise ValueError(f"Failed to generate content: {error_msg}")
        
        if response is None:
            raise ValueError("No response received from Gemini API after retries")
        
        # Check response for errors or blocks
        if not response.candidates:
            logger.error(f"No candidates returned from Gemini for story {story_id}")
            if hasattr(response, 'prompt_feedback'):
                feedback = response.prompt_feedback
                reason = getattr(feedback, 'block_reason', 'Unknown')
                logger.error(f"Block reason: {reason}")
                raise ValueError(f"Content was blocked by safety filters. Reason: {reason}")
            raise ValueError("No response candidates returned from Gemini API.")
        
        # Check finish reason
        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason if hasattr(candidate, 'finish_reason') else None
        
        if finish_reason == 2:  # SAFETY (blocked)
            logger.error(f"Response blocked by safety filters for story {story_id}")
            # Try to get safety ratings
            safety_ratings = getattr(candidate, 'safety_ratings', [])
            blocked_categories = [r.category for r in safety_ratings if hasattr(r, 'category')]
            raise ValueError(
                f"Content was blocked by safety filters. "
                f"Blocked categories: {blocked_categories if blocked_categories else 'Unknown'}. "
                f"Try simplifying the prompt or reducing the context."
            )
        elif finish_reason == 3:  # RECITATION (potential copyright issue)
            logger.warning(f"Response flagged for recitation for story {story_id}")
            # Still try to extract text
        elif finish_reason == 4:  # OTHER
            logger.warning(f"Response finished with reason 'OTHER' for story {story_id}")
        
        # Extract text from response
        result_text = None
        try:
            # Try the quick accessor first
            if hasattr(response, 'text'):
                result_text = response.text
            elif hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                # Extract from parts manually
                parts = candidate.content.parts
                if parts:
                    result_text = parts[0].text if hasattr(parts[0], 'text') else str(parts[0])
        except Exception as e:
            logger.error(f"Error extracting text from response: {str(e)}")
            logger.error(f"Finish reason: {finish_reason}")
            logger.error(f"Response structure: {type(response)}")
            if hasattr(response, 'candidates') and response.candidates:
                logger.error(f"Candidate structure: {type(response.candidates[0])}")
                logger.error(f"Candidate attributes: {dir(response.candidates[0])}")
            raise ValueError(f"Could not extract text from response. Finish reason: {finish_reason}. Error: {str(e)}")
        
        if not result_text:
            logger.error(f"Empty response from Gemini for story {story_id}")
            raise ValueError("Received empty response from Gemini API.")
        
        # Log response preview
        response_preview = result_text[:200] if result_text else "Empty response"
        logger.info(f"Received response with {len(result_text)} characters for story {story_id}")
        logger.debug(f"Response preview (first 200 chars): {response_preview}...")
        
        return result_text
        
    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        logger.error(f"Error generating test cases for story {story_id}: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to generate test cases: {str(e)}")

