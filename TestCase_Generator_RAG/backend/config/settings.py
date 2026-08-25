"""
Configuration settings and environment variable management.
Loads environment variables from .env file and provides centralized configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Required environment variables
_REQUIRED_VARS = [
    "GEMINI_API_KEY",
    "JIRA_URL",
    "JIRA_USER",
    "JIRA_API_TOKEN",
]


def _validate_env_vars():
    """Validate that all required environment variables are present."""
    missing_vars = [var for var in _REQUIRED_VARS if not os.getenv(var)]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}. "
            f"Please ensure these are set in your .env file."
        )


# Validate environment variables on module import
_validate_env_vars()

# Module-level constants
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
JIRA_URL = os.getenv("JIRA_URL")
JIRA_USER = os.getenv("JIRA_USER")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")


def get_gemini_api_key() -> str:
    """
    Get the Gemini API key.
    
    Returns:
        str: The Gemini API key
        
    Raises:
        ValueError: If the API key is not set
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in environment variables")
    return GEMINI_API_KEY

