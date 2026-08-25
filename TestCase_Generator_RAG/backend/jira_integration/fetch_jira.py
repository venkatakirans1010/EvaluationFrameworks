"""
Jira integration module.
Fetches issue details from Jira API using JIRA_URL, JIRA_USER, and JIRA_API_TOKEN.
"""

import re
import requests
from typing import Dict, Any
from base64 import b64encode

from config.settings import JIRA_URL, JIRA_USER, JIRA_API_TOKEN


def fetch_jira_details(story_key: str) -> Dict[str, Any]:
    """
    Fetch Jira issue details including summary, description, acceptance criteria,
    comments, and attachment metadata.
    
    Args:
        story_key: The Jira issue key (e.g., 'PROJ-123')
        
    Returns:
        dict: Dictionary containing:
            - id: Issue key
            - summary: Issue summary
            - description: Issue description (empty string if None)
            - acceptance: Acceptance criteria (empty string if not present)
            - comments: List of comment text strings
            - attachments: List of dicts with 'filename' and 'content_url' keys
            
    Raises:
        ValueError: If the issue is not found (404) or if there's an authentication error
    """
    # Prepare authentication
    auth_string = f"{JIRA_USER}:{JIRA_API_TOKEN}"
    auth_bytes = auth_string.encode('ascii')
    auth_b64 = b64encode(auth_bytes).decode('ascii')
    headers = {
        'Authorization': f'Basic {auth_b64}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    # Construct API URL
    api_url = f"{JIRA_URL}/rest/api/3/issue/{story_key}"
    
    # Fields to expand for full issue details
    params = {
        'fields': 'summary,description,comment,attachment',
        'expand': 'renderedFields'
    }
    
    try:
        # Fetch the issue
        response = requests.get(api_url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 404:
            raise ValueError(f"Jira issue '{story_key}' not found. Please verify the issue key.")
        elif response.status_code == 401:
            raise ValueError(
                f"Authentication failed. Please verify JIRA_USER and JIRA_API_TOKEN "
                f"in your .env file."
            )
        elif not response.ok:
            error_text = response.text
            raise ValueError(f"Error fetching Jira issue '{story_key}': HTTP {response.status_code} - {error_text}")
        
        issue_data = response.json()
        
        # Extract fields
        fields = issue_data.get('fields', {})
        
        # Extract description
        description = ""
        if 'description' in fields:
            desc_content = fields['description']
            if isinstance(desc_content, dict):
                # Handle ADF (Atlassian Document Format) or plain text
                if 'content' in desc_content:
                    # ADF format - extract text
                    description = _extract_text_from_adf(desc_content)
                else:
                    description = str(desc_content)
            elif desc_content:
                description = str(desc_content)
        
        # Extract summary
        summary = fields.get('summary', '') or ""
        
        # Extract acceptance criteria
        acceptance = ""
        # Try common custom field names for acceptance criteria
        common_ac_fields = [
            'customfield_10020', 'customfield_10021', 'customfield_10026',
            'customfield_10027', 'customfield_10028', 'customfield_10029'
        ]
        for field_name in common_ac_fields:
            if field_name in fields and fields[field_name]:
                acceptance = str(fields[field_name])
                break
        
        # If not found in custom fields, try to extract from description
        if not acceptance and description:
            patterns = [
                r'(?i)acceptance\s+criteria[:\s]+\s*(.+?)(?=\n\n|\n[A-Z]|\Z)',
                r'(?i)ac[:\s]+\s*(.+?)(?=\n\n|\n[A-Z]|\Z)',
            ]
            for pattern in patterns:
                match = re.search(pattern, description, re.DOTALL | re.MULTILINE)
                if match:
                    acceptance = match.group(1).strip()
                    break
        
        # Extract comments
        comments = []
        if 'comment' in fields and fields['comment']:
            comment_data = fields['comment']
            if 'comments' in comment_data:
                for comment in comment_data['comments']:
                    comment_body = comment.get('body', '')
                    if isinstance(comment_body, dict):
                        comment_text = _extract_text_from_adf(comment_body)
                    else:
                        comment_text = str(comment_body)
                    if comment_text:
                        comments.append(comment_text)
        
        # Extract attachment metadata
        attachments = []
        if 'attachment' in fields and fields['attachment']:
            for attachment in fields['attachment']:
                attachments.append({
                    'filename': attachment.get('filename', 'Unknown'),
                    'content_url': attachment.get('content', '')
                })
        
        return {
            'id': issue_data.get('key', story_key),
            'summary': summary,
            'description': description,
            'acceptance': acceptance,
            'comments': comments,
            'attachments': attachments
        }
        
    except ValueError:
        # Re-raise ValueError as-is
        raise
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Network error fetching Jira issue '{story_key}': {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error fetching Jira issue '{story_key}': {str(e)}")


def _extract_text_from_adf(adf_content: dict) -> str:
    """
    Extract plain text from Atlassian Document Format (ADF).
    
    Args:
        adf_content: ADF content dictionary
        
    Returns:
        Plain text string
    """
    text_parts = []
    
    if isinstance(adf_content, dict):
        if 'content' in adf_content:
            for item in adf_content.get('content', []):
                if item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif item.get('type') == 'paragraph':
                    text_parts.append(_extract_text_from_adf(item))
                elif 'content' in item:
                    text_parts.append(_extract_text_from_adf(item))
        elif 'text' in adf_content:
            text_parts.append(str(adf_content['text']))
    
    return '\n'.join(text_parts).strip()


