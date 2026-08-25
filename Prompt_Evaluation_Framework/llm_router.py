"""
LLM Router Module
Handles routing prompts to different LLM providers using RouteLLM/AbacusAI or direct APIs
"""
import requests
import json
from typing import Dict, List, Optional, Any
import time

class LLMRouter:
    """Router for managing multiple LLM API calls"""
    
    def __init__(self):
        # RouteLLM API base URL
        self.routellm_base_url = "https://routellm.abacus.ai/v1"
        self.responses = []
    
    def call_routellm(
        self,
        prompt: str,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call RouteLLM API (AbacusAI unified API)
        
        Args:
            prompt: The input prompt
            model: Model name (e.g., 'gpt-4', 'claude-3-opus', 'llama-3-70b')
            api_key: RouteLLM API key
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with response data
        """
        url = f"{self.routellm_base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            # Extract response text
            response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Debug: Log response structure if empty
            if not response_text:
                print(f"DEBUG: Empty response from API. Full response: {json.dumps(data, indent=2)[:500]}")
            
            return {
                "success": True,
                "model": model,
                "response": response_text,
                "usage": data.get("usage", {}),
                "provider": "RouteLLM",
                "error": None,
                "debug_info": {
                    "response_structure": list(data.keys()) if isinstance(data, dict) else "Not a dict",
                    "choices_count": len(data.get("choices", [])) if isinstance(data, dict) else 0
                } if not response_text else None
            }
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f" | Status: {e.response.status_code} | Response: {e.response.text[:200]}"
            return {
                "success": False,
                "model": model,
                "response": None,
                "usage": None,
                "provider": "RouteLLM",
                "error": error_msg
            }
    
    def call_openai(
        self,
        prompt: str,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Call OpenAI API directly"""
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            return {
                "success": True,
                "model": model,
                "response": data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "usage": data.get("usage", {}),
                "provider": "OpenAI",
                "error": None
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "model": model,
                "response": None,
                "usage": None,
                "provider": "OpenAI",
                "error": str(e)
            }
    
    def call_anthropic(
        self,
        prompt: str,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Call Anthropic API directly"""
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            return {
                "success": True,
                "model": model,
                "response": data.get("content", [{}])[0].get("text", ""),
                "usage": data.get("usage", {}),
                "provider": "Anthropic",
                "error": None
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "model": model,
                "response": None,
                "usage": None,
                "provider": "Anthropic",
                "error": str(e)
            }
    
    def call_with_vision(
        self,
        content: List[Dict[str, Any]],
        model: str,
        provider: str,
        api_key: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 4000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call LLM API with vision/image support
        
        Args:
            content: List of content items, each with 'type' ('text' or 'image_url') and content
            model: Model name
            provider: Provider name ('routellm', 'openai', 'anthropic')
            api_key: API key
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with response data
        """
        if provider == "openai":
            return self._call_openai_vision(content, model, api_key, temperature, top_p, max_tokens, **kwargs)
        elif provider == "anthropic":
            return self._call_anthropic_vision(content, model, api_key, temperature, top_p, max_tokens, **kwargs)
        elif provider == "routellm":
            return self._call_routellm_vision(content, model, api_key, temperature, top_p, max_tokens, **kwargs)
        else:
            return {
                "success": False,
                "model": model,
                "response": None,
                "usage": None,
                "provider": provider,
                "error": f"Vision not supported for provider: {provider}"
            }
    
    def _call_openai_vision(
        self,
        content: List[Dict[str, Any]],
        model: str,
        api_key: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Call OpenAI API with vision support"""
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": content}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            
            return {
                "success": True,
                "model": model,
                "response": data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "usage": data.get("usage", {}),
                "provider": "OpenAI",
                "error": None
            }
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f" | Status: {e.response.status_code} | Response: {e.response.text[:200]}"
            return {
                "success": False,
                "model": model,
                "response": None,
                "usage": None,
                "provider": "OpenAI",
                "error": error_msg
            }
    
    def _call_anthropic_vision(
        self,
        content: List[Dict[str, Any]],
        model: str,
        api_key: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Call Anthropic API with vision support"""
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        # Convert content format for Anthropic
        anthropic_content = []
        for item in content:
            if item.get("type") == "text":
                anthropic_content.append({"type": "text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {}).get("url", "")
                # Extract base64 data
                if image_url.startswith("data:"):
                    # Format: data:image/png;base64,<data>
                    parts = image_url.split(",", 1)
                    if len(parts) == 2:
                        mime_type = parts[0].split(":")[1].split(";")[0]
                        base64_data = parts[1]
                        anthropic_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": base64_data
                            }
                        })
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": anthropic_content}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            
            return {
                "success": True,
                "model": model,
                "response": data.get("content", [{}])[0].get("text", ""),
                "usage": data.get("usage", {}),
                "provider": "Anthropic",
                "error": None
            }
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f" | Status: {e.response.status_code} | Response: {e.response.text[:200]}"
            return {
                "success": False,
                "model": model,
                "response": None,
                "usage": None,
                "provider": "Anthropic",
                "error": error_msg
            }
    
    def _call_routellm_vision(
        self,
        content: List[Dict[str, Any]],
        model: str,
        api_key: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Call RouteLLM API with vision support (OpenAI-compatible format)"""
        url = f"{self.routellm_base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": content}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            
            response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            return {
                "success": True,
                "model": model,
                "response": response_text,
                "usage": data.get("usage", {}),
                "provider": "RouteLLM",
                "error": None
            }
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f" | Status: {e.response.status_code} | Response: {e.response.text[:200]}"
            return {
                "success": False,
                "model": model,
                "response": None,
                "usage": None,
                "provider": "RouteLLM",
                "error": error_msg
            }
    
    def route_to_models(
        self,
        prompt: str,
        model_configs: List[Dict[str, Any]],
        api_keys: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Route a prompt to multiple models concurrently
        
        Args:
            prompt: The input prompt
            model_configs: List of model configurations, each containing:
                - provider: 'routellm', 'openai', 'anthropic'
                - model: model name
                - temperature, top_p, max_tokens, etc.
            api_keys: Dictionary mapping provider names to API keys
        
        Returns:
            List of response dictionaries
        """
        results = []
        
        for config in model_configs:
            provider = config.get("provider", "routellm").lower()
            model = config.get("model", "")
            api_key = api_keys.get(provider, "")
            
            if not api_key:
                results.append({
                    "success": False,
                    "model": model,
                    "response": None,
                    "usage": None,
                    "provider": provider,
                    "error": f"No API key provided for {provider}"
                })
                continue
            
            # Extract parameters
            temperature = config.get("temperature", 0.7)
            top_p = config.get("top_p", 1.0)
            max_tokens = config.get("max_tokens", 1000)
            
            # Route to appropriate provider
            if provider == "routellm":
                result = self.call_routellm(
                    prompt, model, api_key, temperature, top_p, max_tokens
                )
            elif provider == "openai":
                result = self.call_openai(
                    prompt, model, api_key, temperature, top_p, max_tokens
                )
            elif provider == "anthropic":
                result = self.call_anthropic(
                    prompt, model, api_key, temperature, top_p, max_tokens
                )
            else:
                result = {
                    "success": False,
                    "model": model,
                    "response": None,
                    "usage": None,
                    "provider": provider,
                    "error": f"Unsupported provider: {provider}"
                }
            
            results.append(result)
            time.sleep(0.1)  # Small delay to avoid rate limits
        
        return results


