import httpx
import asyncio
import logging
import os
from typing import List, Dict, Any, Optional, AsyncGenerator
import json

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, host: str = None, model_name: str = None):
        """Initialize LLM service with Ollama."""
        self.host = host or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.model_name = model_name or os.getenv('OLLAMA_MODEL', 'llama3.2:1b')  # Smaller model
        self.timeout = float(os.getenv('OLLAMA_TIMEOUT', '120.0'))  # Increased to 2 minutes
        self.max_context_length = int(os.getenv('MAX_CONTEXT_LENGTH', '2000'))  # Reduced context
        logger.info(f"Initialized LLM service with model: {self.model_name} at {self.host}")

    async def generate_response(self,
                              prompt: str,
                              context_docs: List[Dict[str, Any]] = None,
                              system_prompt: str = None,
                              temperature: float = 0.7,
                              max_tokens: int = 1000) -> str:  # Reduced max tokens
        """Generate response using Ollama model with RAG context."""
        try:
            # Build prompt with context length optimization
            full_prompt = self._build_rag_prompt(prompt, context_docs, system_prompt)
            
            # Truncate if too long
            if len(full_prompt) > self.max_context_length:
                logger.warning(f"Prompt too long ({len(full_prompt)}), truncating to {self.max_context_length}")
                full_prompt = full_prompt[:self.max_context_length] + "..."
            
            logger.info(f"Sending request to Ollama at {self.host} (model: {self.model_name})")
            logger.info(f"Prompt length: {len(full_prompt)} characters")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Try the generate endpoint first
                try:
                    response = await client.post(
                        f"{self.host}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": full_prompt,
                            "stream": False,
                            "options": {
                                "temperature": temperature,
                                "num_predict": max_tokens,
                                "stop": ["Human:", "User:", "\n\nUser:", "\n\nHuman:"],
                                "num_ctx": 2048  # Limit context window
                            }
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    result = data.get('response', '').strip()
                    
                    if result:
                        logger.info(f"Received response from Ollama: {result[:100]}...")
                        return result
                    else:
                        logger.warning("Empty response from Ollama")
                        return "I'm sorry, I couldn't generate a response."
                        
                except Exception as e:
                    logger.error(f"Generate API failed: {e}")
                    # Fallback to chat API
                    return await self._try_chat_api(client, full_prompt, temperature, max_tokens)
                
        except httpx.TimeoutException:
            logger.error(f"Ollama request timed out after {self.timeout} seconds")
            return self._get_fallback_response(prompt, context_docs)
        except httpx.ConnectError:
            logger.error(f"Failed to connect to Ollama at {self.host}")
            return "I'm sorry, I cannot connect to the AI service. Please ensure Ollama is running."
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return self._get_fallback_response(prompt, context_docs)

    async def _try_chat_api(self, client, prompt, temperature, max_tokens):
        """Fallback to chat API if generate fails."""
        try:
            response = await client.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "num_ctx": 2048
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get('message', {}).get('content', '').strip()
        except Exception as e:
            logger.error(f"Chat API also failed: {e}")
            raise

    def _get_fallback_response(self, prompt: str, context_docs: List[Dict[str, Any]] = None) -> str:
        """Generate a fallback response when Ollama fails."""
        if context_docs:
            # Try to extract useful information from context
            weather_info = []
            for doc in context_docs:
                content = doc.get('content', '')
                if 'temperature' in content.lower() or 'weather' in content.lower():
                    # Extract first few sentences that might contain weather info
                    sentences = content.split('.')[:3]
                    weather_info.extend(sentences)
            
            if weather_info:
                return f"Based on the available information: {' '.join(weather_info[:2])}. Please note that this information may not be the most current. For the latest weather updates, please check a reliable weather service."
        
        if 'weather' in prompt.lower():
            return "I'm experiencing technical difficulties accessing the AI service. For current weather information, I recommend checking a reliable weather service like weather.com or your local weather app."
        
        return "I'm sorry, I'm experiencing technical difficulties. Please try again in a moment or rephrase your question."

    async def stream_response(self,
                            prompt: str,
                            context_docs: List[Dict[str, Any]] = None,
                            system_prompt: str = None,
                            temperature: float = 0.7) -> AsyncGenerator[str, None]:
        """Stream response from Ollama model."""
        try:
            full_prompt = self._build_rag_prompt(prompt, context_docs, system_prompt)
            
            # Truncate if too long
            if len(full_prompt) > self.max_context_length:
                full_prompt = full_prompt[:self.max_context_length] + "..."
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    'POST',
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": full_prompt,
                        "stream": True,
                        "options": {
                            "temperature": temperature,
                            "num_ctx": 2048
                        }
                    }
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if 'response' in data and data['response']:
                                    yield data['response']
                                if data.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield "Error occurred while streaming response."

    def _build_rag_prompt(self,
                         user_query: str,
                         context_docs: List[Dict[str, Any]] = None,
                         system_prompt: str = None) -> str:
        """Build a complete prompt with RAG context (optimized for length)."""
        if not system_prompt:
            system_prompt = "You are a helpful AI assistant. Answer questions accurately and concisely based on the provided context."

        prompt_parts = [system_prompt]
        
        if context_docs:
            prompt_parts.append("\n--- CONTEXT ---")
            total_context_length = 0
            max_context_per_doc = 500  # Limit each document to 500 chars
            
            for i, doc in enumerate(context_docs[:3]):  # Limit to 3 docs
                content = doc.get('content', '')
                source = doc.get('metadata', {}).get('filename', 'Web')
                
                # Truncate long content
                if len(content) > max_context_per_doc:
                    content = content[:max_context_per_doc] + "..."
                
                total_context_length += len(content)
                if total_context_length > 1200:  # Limit total context
                    break
                    
                prompt_parts.append(f"\nSource {i+1} ({source}): {content}")
            
            prompt_parts.append("\n--- END CONTEXT ---")
        
        prompt_parts.append(f"\nQuestion: {user_query}")
        prompt_parts.append("\nAnswer:")
        
        return "\n".join(prompt_parts)

    async def check_model_availability(self) -> bool:
        """Check if the specified model is available."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.host}/api/tags")
                response.raise_for_status()
                data = response.json()
                
                available_models = []
                if 'models' in data:
                    available_models = [model.get("name", "") for model in data['models']]
                
                is_available = self.model_name in available_models
                if not is_available:
                    logger.warning(f"Model {self.model_name} not found. Available models: {available_models}")
                    # Try to use a smaller model if available
                    smaller_models = ['llama3.2:1b', 'llama2:7b', 'phi3:mini', 'gemma:2b']
                    for model in smaller_models:
                        if model in available_models:
                            logger.info(f"Switching to smaller model: {model}")
                            self.model_name = model
                            return True
                
                return is_available
                
        except Exception as e:
            logger.error(f"Model check error: {e}")
            return False

    async def pull_model(self) -> bool:
        """Pull/download the model if not available.""" 
        try:
            logger.info(f"Pulling model: {self.model_name}")
            async with httpx.AsyncClient(timeout=600.0) as client:  # 10 minute timeout for model pull
                response = await client.post(
                    f"{self.host}/api/pull",
                    json={"name": self.model_name}
                )
                response.raise_for_status()
                logger.info(f"Successfully pulled model: {self.model_name}")
                return True
        except Exception as e:
            logger.error(f"Model pull error: {e}")
            return False

    async def list_available_models(self) -> List[str]:
        """List all available Ollama models."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.host}/api/tags")
                response.raise_for_status()
                data = response.json()
                
                model_list = []
                if 'models' in data:
                    model_list = [m.get('name', '') for m in data['models']]
                
                return [name for name in model_list if name]
        except Exception as e:
            logger.error(f"List models error: {e}")
            return []

    def set_model(self, model_name: str):
        """Change the active model."""
        self.model_name = model_name
        logger.info(f"Changed model to: {model_name}")

    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Summarize long text using the LLM."""
        summary_prompt = f"Summarize this text in {max_length} words or less:\n\n{text[:1000]}\n\nSummary:"
        return await self.generate_response(
            summary_prompt,
            temperature=0.3,
            max_tokens=max_length + 50
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "host": self.host,
            "status": "connected",
            "timeout": self.timeout,
            "max_context_length": self.max_context_length
        }

    async def close(self):
        """Close any open connections."""
        pass
